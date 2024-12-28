import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from gymnasium.spaces import Box, Discrete
from gymnasium import Env
import pygame
from .config import *


class DefenderEnv(Env):
    def __init__(self, game):
        super().__init__()
        self.game = game

        # Пространство состояний (непрерывное)
        self.observation_space = Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)

        # Расширенное пространство действий
        self.action_space = Discrete(ACTION_DIM)  # Влево, Вправо, Вверх, Вниз, Стрельба

    def step(self, action, visualize=False, training=False):
        state, score, done = self.game.step(action, visualize, training)
        return state, score, done, False, {}

    def reset(self, seed=None, training=False):
        super().reset(seed=seed)
        return self.game.reset(training), {}


class DefenderDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DefenderDQNNetwork, self).__init__()
        
        # Расширенная архитектура сети
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Advantage stream
        self.advantage_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Value stream
        self.value_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.feature_net(x)
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        
        # Dueling architecture combination
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.priorities = []
        self.max_size = max_size
        self.alpha = 0.6  # Приоритет
        self.beta = 0.4  # Важность выборки
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Малое число для избежания нулевого приоритета

    def add(self, state, action, score, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) >= self.max_size:
            # Оптимизированное удаление старых элементов
            self.buffer = self.buffer[-self.max_size:]
            self.priorities = self.priorities[-self.max_size:]

        # Проверяем, что состояния уже являются numpy массивами нужной формы
        if not isinstance(state, np.ndarray) or state.shape != (STATE_DIM,):
            return
        if not isinstance(next_state, np.ndarray) or next_state.shape != (STATE_DIM,):
            return
            
        # Создаем копии массивов
        transition = (
            state.copy(),  # Уже проверили, что это numpy массив нужной формы
            int(action),
            float(score),
            next_state.copy(),  # Уже проверили, что это numpy массив нужной формы
            bool(done)
        )
        
        self.buffer.append(transition)
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        # Проверяем, что у нас достаточно опыта
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            batch_size = buffer_len

        # Нормализуем приоритеты
        total = np.sum([p ** self.alpha for p in self.priorities[:buffer_len]])
        if total == 0:
            probabilities = np.ones(buffer_len) / buffer_len
        else:
            probabilities = np.array([p ** self.alpha / total for p in self.priorities[:buffer_len]])

        # Выбираем индексы на основе приоритетов
        indices = np.random.choice(buffer_len, batch_size, p=probabilities).astype(np.int32)
        samples = [self.buffer[int(idx)] for idx in indices]

        # Ensure scores are scalar values
        scores = [score[0] if isinstance(score, (list, np.ndarray)) and len(score) == 1 else score for score in [sample[2] for sample in samples]]
        
        # Convert to numpy arrays
        states_array = np.stack([np.array(sample[0], dtype=np.float32) for sample in samples])
        next_states_array = np.stack([np.array(sample[3], dtype=np.float32) for sample in samples])
        actions_array = np.array([sample[1] for sample in samples], dtype=np.int64)
        scores_array = np.array(scores, dtype=np.float32)
        dones_array = np.array([sample[4] for sample in samples], dtype=np.float32)
        weights_array = np.array([(buffer_len * probabilities[int(idx)]) ** (-self.beta) / max((buffer_len * probabilities[int(idx)]) ** (-self.beta) for idx in indices) for idx in indices], dtype=np.float32)

        return (states_array, actions_array, scores_array, 
                next_states_array, dones_array, weights_array, indices)

    def update_priorities(self, indices, errors):
        # Безопасное обновление приоритетов
        for idx, error in zip(indices, errors):
            idx = int(idx)  # Преобразуем индекс в целое число
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(abs(error)) + self.epsilon
            else:
                print(f"Warning: Invalid index {idx} for priority update")


class DefenderDQNAgent:
    def __init__(self, game,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON_START,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_min=EPSILON_MIN,
                 tau=TAU,
                 batch_size=BATCH_SIZE):

        self.game = game

        # Параметры обучения
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau

        # Размерности состояний и действий из конфига
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

        # Улучшенные нейронные сети
        self.policy_network = DefenderDQNNetwork(self.state_dim, self.action_dim)
        self.target_network = DefenderDQNNetwork(self.state_dim, self.action_dim)

        # Копирование весов
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Оптимизатор с адаптивным learning rate и L2 регуляризацией
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            eps=1e-4,
            weight_decay=1e-5
        )

        # Функция потерь с уменьшением разброса
        self.loss_fn = nn.SmoothL1Loss()

        # Приоритетный буфер воспроизведения
        self.replay_buffer = PrioritizedReplayBuffer(MAX_BUFFER_SIZE)
        self.batch_size = batch_size

        # История обучения
        self.training_history = []
        self.loss_history = []
        self.accuracy_history = []
        self.td_error_history = []
        self.q_value_history = []

        # Переменные для отслеживания лучшей модели
        self.best_score = float('-inf')
        self.save_interval = 10  # Сохранять модель каждые N эпизодов

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Увеличиваем вероятность выбора стрельбы при случайном действии
            if random.random() < 0.45:  # Увеличена вероятность случайной стрельбы до 45%
                return SHOOT
            # Увеличиваем вероятность осмысленных движений
            if random.random() < 0.7:  # 70% шанс выбрать движение вместо стрельбы
                # Предпочитаем движения влево/вправо и вверх/вниз
                return random.choice([MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN])
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_network(state_tensor)
            
            # Увеличиваем бонус для стрельбы
            q_values[0][MOVE_LEFT] += 0.05
            q_values[0][MOVE_RIGHT] += 0.05
            q_values[0][MOVE_UP] += 0.05
            q_values[0][MOVE_DOWN] += 0.05
            q_values[0][SHOOT] += 0.25 

            return q_values.argmax().item()

    def store_transition(self, state, action, score, next_state, done):
        # Убеждаемся, что состояния являются numpy массивами
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        # Проверяем размерность
        if state.size != STATE_DIM:
            return
        if next_state.size != STATE_DIM:
            return
            
        # Приводим к нужной форме
        state = state.reshape(STATE_DIM)
        next_state = next_state.reshape(STATE_DIM)
            
        # Ensure score is a sequence
        if not isinstance(score, (list, np.ndarray)):
            score = [score]
        
        # Normalize score for stability
        normalized_score = np.clip(score, -20, 20)  # Expanded normalization range

        self.replay_buffer.add(state, action, normalized_score, next_state, done)

    def train(self, total_episodes=1000, visualize=False, progress_callback=None):
        import gc
        gc_counter = 0
        
        # Создаем директории для сохранения результатов
        os.makedirs('results/logs', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/training_plots', exist_ok=True)

        with open('results/logs/training_log.txt', 'w') as log_file:
            log_file.write("DQN Training Log\n")
            log_file.write(f"Total Episodes: {total_episodes}\n")
            log_file.write(f"Learning Rate: {self.learning_rate}\n")
            log_file.write(f"Gamma: {self.gamma}\n")
            log_file.write(f"Initial Epsilon: {self.epsilon}\n")
            log_file.write(f"Epsilon Decay: {self.epsilon_decay}\n")
            log_file.write(f"Minimum Epsilon: {self.epsilon_min}\n")
            log_file.write(f"Batch Size: {self.batch_size}\n")
            log_file.write("=" * 140 + "\n\n")
            log_file.write(f"{'Episode':<10}{'Score':<13}{'Humans':<8}{'Meteors':<10}"
                          f"{'Rewards':<13}{'Penalties':<13}{'Loss':<16}{'TD Error':<16}"
                          f"{'Avg Q':<14}{'Accuracy':<14}{'Epsilon':<8}\n")

            for episode in range(total_episodes):
                state = self.game.reset(training=True)[0]
                done = False
                total_score = 0
                episode_losses = []
                episode_td_errors = []
                episode_q_values = []
                meteors_destroyed = 0
                humans_saved = 0
                total_rewards = 0
                total_penalties = 0
                step_count = 0
                successful_actions = 0
                total_actions = 0
                shots_fired = 0
                shots_hit = 0

                while not done:
                    action = self.select_action(state)
                    next_state, score, done = self.game.step(action, visualize, training=True)

                    # Увеличиваем счетчик всех действий
                    total_actions += 1

                    # Подсчет успешных действий
                    if hasattr(self.game, 'last_destroyed_meteors'):
                        meteors_destroyed += len(self.game.last_destroyed_meteors)
                        shots_hit += len(self.game.last_destroyed_meteors)
                    if hasattr(self.game, 'last_saved_humans'):
                        humans_saved += len(self.game.last_saved_humans)
                        successful_actions += len(self.game.last_saved_humans)

                    # Подсчет выстрелов
                    if action == SHOOT:
                        shots_fired += 1
                    
                    if score > 0:
                        total_rewards += score
                        successful_actions += 1
                    else:
                        total_penalties += abs(score)

                    normalized_score = score / (step_count + 1)
                    self.store_transition(state, action, normalized_score, next_state, done)
                    state = next_state
                    total_score += score
                    step_count += 1

                    if len(self.replay_buffer.buffer) >= self.batch_size:
                        loss_info = self.replay()
                        if loss_info is not None:
                            episode_losses.append(loss_info[0])
                            episode_td_errors.append(loss_info[1])
                            episode_q_values.append(loss_info[2])

                    self.softupdate_target_network()

                # Вычисляем средние значения для эпизода
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
                avg_q = np.mean(episode_q_values) if episode_q_values else 0

                # Расчет точности с учетом разных метрик
                if total_actions == 0:
                    accuracy = 0.0
                else:
                    # Эффективность действий
                    action_efficiency = successful_actions / total_actions
                    
                    # Эффективность стрельбы (если были выстрелы)
                    if shots_fired > 0:
                        shooting_accuracy = shots_hit / shots_fired
                    else:
                        shooting_accuracy = 0.0
                    
                    # Эффективность выживания
                    survival_score = max(0, (step_count - total_penalties) / step_count) if step_count > 0 else 0
                    
                    # Эффективность защиты людей
                    if self.game.total_humans_appeared > 0:
                        human_protection = humans_saved / self.game.total_humans_appeared
                    else:
                        human_protection = 1.0  # Если людей не было, считаем что защита была успешной
                    
                    # Эффективность уничтожения метеоритов
                    if self.game.total_meteors_appeared > 0:
                        meteor_efficiency = meteors_destroyed / self.game.total_meteors_appeared
                    else:
                        meteor_efficiency = 1.0  # Если метеоритов не было, считаем что защита была успешной
                    
                    # Общая точность как взвешенная сумма всех метрик
                    accuracy = np.clip(
                        action_efficiency * 0.2 +      # Общая эффективность действий
                        shooting_accuracy * 0.3 +      # Точность стрельбы
                        survival_score * 0.1 +         # Выживаемость
                        human_protection * 0.2 +       # Защита людей
                        meteor_efficiency * 0.2,       # Уничтожение метеоритов
                        0.0, 1.0
                    )

                # Сохраняем историю без ограничений
                self.training_history.append(total_score)
                self.loss_history.append(avg_loss)
                self.accuracy_history.append(accuracy)
                self.td_error_history.append(avg_td_error)
                self.q_value_history.append(avg_q)

                # Логирование результатов эпизода
                log_file.write(
                    f"{episode:<10}{total_score:<13.2f}{humans_saved:<8}{meteors_destroyed:<10}"
                    f"{total_rewards:<13.2f}{total_penalties:<13.2f}{avg_loss:<16.8f}{avg_td_error:<16.8f}"
                    f"{avg_q:<14.8f}{accuracy:<14.8f}{self.epsilon:<8.4f}\n"
                )
                log_file.flush()

                # Сохраняем модель только каждые N эпизодов
                if episode % self.save_interval == 0:
                    self.save_model(f"results/models/defender_dqn_last_episode")

                if total_score > self.best_score:
                    self.best_score = total_score
                    self.save_model("results/models/defender_dqn_best")

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                # Каждые 10 эпизодов выполняем сборку мусора
                gc_counter += 1
                if gc_counter >= 10:
                    gc.collect()  # Принудительная сборка мусора
                    gc_counter = 0
                    
                # Очищаем кэши игры
                self.game.scaled_meteor_images.clear()
                self.game.cached_texts.clear()

                if progress_callback:
                    progress_callback(episode)

            # Финальная статистика
            log_file.write("\n")
            log_file.write("=" * 140 + "\n")
            log_file.write("Final Training Statistics\n")
            log_file.write(f"Total Episodes: {total_episodes}\n")
            log_file.write(f"Best score: {self.best_score:.2f}\n")

            # Финальное сохранение
            self.save_model("results/models/defender_dqn_final")
            self.plot_training_history()
            self.plot_loss_history()
            self.plot_accuracy_history()
            self.plot_td_error_history()
            self.plot_q_value_history()

    def softupdate_target_network(self):
        # Мягкое обновление целевой сети
        for target_param, policy_param in zip(
                self.target_network.parameters(),
                self.policy_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data +
                (1.0 - self.tau) * target_param.data
            )

    def replay(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return 0, 0, 0
            
        # Получаем batch с приоритетами
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
        
        # Конвертация в тензоры
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        # Double DQN: выбор действий через policy network
        with torch.no_grad():
            next_q_values = self.policy_network(next_states)
            next_actions = next_q_values.argmax(1)
            
            # Получение Q-значений через target network для выбранных действий
            next_target_q_values = self.target_network(next_states)
            next_target_q_values = next_target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Вычисление целевых Q-значений с учетом done
            target_q_values = rewards + (1 - dones) * self.gamma * next_target_q_values
        
        # Получение текущих Q-значений
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Вычисление TD-ошибок и потерь с учетом весов
        td_errors = torch.abs(target_q_values - current_q_values)
        
        # Масштабируем TD-ошибки для более реалистичных значений потерь
        scaled_td_errors = td_errors * 10.0
        
        # Используем Huber Loss с увеличенным порогом
        losses = torch.where(scaled_td_errors < 5.0,
                           0.5 * scaled_td_errors.pow(2),
                           5.0 * scaled_td_errors - 12.5)
        
        # Применяем веса из приоритетного буфера
        weighted_losses = losses * weights
        loss = weighted_losses.mean()
        
        # Обновление приоритетов в буфере (используем немасштабированные ошибки)
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().numpy()
        else:
            td_errors = np.array([float(td_errors)])
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        return (
            loss.item(),
            td_errors.mean().item(),
            current_q_values.mean().item()
        )

    def update_target_network(self):
        # Обновление весов целевой сети
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model(self, path):
        # Сохранение только весов модели
        torch.save({
            'policy_network_state': self.policy_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path, _use_new_zipfile_serialization=True)

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_history)
        plt.title('Счет во время обучения DQN')
        plt.xlabel('Эпизод')
        plt.ylabel('Общий счет')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/training_plots/dqn_training_scores.png')
        plt.close()

    def plot_loss_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title('Средняя функция потерь за эпизод')
        plt.xlabel('Эпизод')
        plt.ylabel('Среднее значение потерь')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/training_plots/dqn_training_loss.png')
        plt.close()

    def plot_accuracy_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.accuracy_history)
        plt.title('Точность модели во время обучения')
        plt.xlabel('Эпизод')
        plt.ylabel('Точность (0-1)')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/training_plots/dqn_training_accuracy.png')
        plt.close()

    def plot_td_error_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.td_error_history)
        plt.title('Среднее TD Error за эпизод')
        plt.xlabel('Эпизод')
        plt.ylabel('TD Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/training_plots/dqn_training_td_error.png')
        plt.close()

    def plot_q_value_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.q_value_history)
        plt.title('Средние Q-значения за эпизод')
        plt.xlabel('Эпизод')
        plt.ylabel('Среднее Q-значение')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/training_plots/dqn_training_q_values.png')
        plt.close()

    def init_networks(self):
        # Метод для повторной инициализации сетей
        self.policy_network = DefenderDQNNetwork(self.state_dim, self.action_dim)
        self.target_network = DefenderDQNNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Переинициализация оптимизатора
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            eps=1e-4
        )

    def load_model(self, path):
        try:
            # Проверка существования файла
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели не найден: {path}")

            # Загрузка с явным указанием weights_only
            checkpoint = torch.load(path, weights_only=True)

            # Загрузка весов
            self.policy_network.load_state_dict(checkpoint['policy_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])

            # Восстановление состояния оптимизатора
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # Восстановление epsilon, если есть
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']

            # Синхронизация целевой сети
            self.target_network.load_state_dict(self.policy_network.state_dict())

            print(f"Модель успешно загружена из {path}")

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            # Инициализация с нуля в случае ошибки
            self.init_networks()

    def play(self, render=True, infinite_health=False):
        # Установка бесконечного здоровья перед началом игры
        self.game.infinite_health = infinite_health

        # Инициализация pygame и сброс состояния игры
        self.game.init_pygame()
        state = self.game.reset()
        total_score = 0
        clock = pygame.time.Clock()
        running = True

        while running:
            # Обработка событий pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game.game_over:
                        # Перезапуск игры по нажатию пробела
                        state = self.game.reset()
                        total_score = 0
                        continue

            if not self.game.game_over:
                # Выбор действия и выполнение шага только если игра не окончена
                action = self.select_action(state)
                state, score, done = self.game.step(action)
                total_score += score

            if render:
                self.game.render()
                clock.tick(FPS)

        pygame.quit()

    def load_training_data(self):
        """Load training data from the log file."""
        try:
            log_path = 'results/logs/training_log.txt'
            if not os.path.exists(log_path):
                print(f"Training log file not found: {log_path}")
                return False

            self.training_history = []
            self.loss_history = []
            self.accuracy_history = []
            self.td_error_history = []
            self.q_value_history = []

            with open(log_path, 'r') as log_file:
                # Skip header lines including the column names
                for _ in range(10):
                    next(log_file)
                
                # Read and parse data lines
                for line in log_file:
                    if line.strip() and not line.startswith('='):
                        try:
                            data = line.split()
                            if len(data) >= 11:  # Make sure we have all columns
                                self.training_history.append(float(data[1]))  # Score
                                self.loss_history.append(float(data[6]))      # Loss
                                self.accuracy_history.append(float(data[9]))  # Accuracy
                                self.td_error_history.append(float(data[7]))  # TD Error
                                self.q_value_history.append(float(data[8]))   # Avg Q
                        except (ValueError, IndexError) as e:
                            print(f"Skipping invalid line: {line.strip()}")
                            continue
            
            if not self.training_history:
                print("No valid training data found in log file")
                return False
                
            return True
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False

    def generate_all_plots(self):
        if not any([self.training_history, self.loss_history, self.accuracy_history, 
                   self.td_error_history, self.q_value_history]):
            if not self.load_training_data():
                return False
        
        # Create plots directory if it doesn't exist
        os.makedirs('results/training_plots', exist_ok=True)
        
        # Generate all plots
        self.plot_training_history()
        self.plot_loss_history()
        self.plot_accuracy_history()
        self.plot_td_error_history()
        self.plot_q_value_history()
        return True
