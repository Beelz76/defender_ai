import pygame
import random
import numpy as np
from .config import *
import os


class Defender:
    def __init__(self):
        self.screen = None
        self.clock = pygame.time.Clock()
        self.last_update_time = pygame.time.get_ticks()
        self.delta_time = 0

        # Флаг бесконечного здоровья по умолчанию
        self.infinite_health = False

        # Система жизней для реальной игры
        self.lives = 3
        self.game_over = False
        self.total_score = 0
        self.total_saved_humans = 0
        self.destroyed_meteors = 0  # Добавляем счетчик уничтоженных метеоритов
        self.last_saved_humans = []  # Добавляем список для отслеживания спасенных людей за шаг

        # Счетчики для общего количества появившихся объектов
        self.total_humans_appeared = 0
        self.total_meteors_appeared = 0
        self.last_destroyed_meteors = []  # Список уничтоженных метеоритов за последний шаг

        # Флаг режима обучения
        self.training_mode = False

        # Загрузка ассетов
        self.ship_img = None
        self.meteor_img = None
        self.human_img = None
        self.background_img = None
        self.bullet_img = None

        # Новые атрибуты для плавного движения
        self.ship_velocity_x = 0
        self.ship_velocity_y = 0

        # Добавляем атрибуты для контроля стрельбы
        self.last_shot_time = 0
        self.shooting_cooldown = BULLET_COOLDOWN

        # Новые атрибуты для системы спавна
        self.last_meteor_spawn = 0
        self.last_human_spawn = 0
        self.meteor_wave_timer = 0
        self.wave_count = 0
        self.meteors_in_wave = 0
        self.max_meteors_per_wave = WAVE_INITIAL_METEORS
        self.wave_duration = WAVE_DURATION
        self.min_spawn_distance = MIN_SPAWN_DISTANCE  # Минимальное расстояние между объектами

        # Атрибуты для отслеживания наград и штрафов
        self.reward_info = {
            'saved': 0,
            'destroyed': 0,
            'near_human': 0,
            'near_meteor': 0,
            'meteor_dodge': 0,
            'edge_penalty': 0,
            'accuracy_penalty': 0,
            'damage_penalty': 0,
            'accuracy': 0,  # Награда за точность
            'protection': 0,  # Награда за защиту людей
        }

        # Список для хранения деталей уничтоженных метеоритов
        self.last_meteor_details = []

        # Сброс игры
        self.reset()

        # Добавляем отслеживание предыдущих позиций
        self.position_history = []
        self.max_history = 50  # Хранить последние 50 позиций
        self.last_movement_time = 0

        # Кэш для масштабированных изображений метеоритов
        self.scaled_meteor_images = {}
        # Кэш для текстов HUD
        self.cached_texts = {}

    def init_pygame(self):
        if not pygame.get_init():  # Проверяем, инициализирован ли pygame
            pygame.init()
            # Инициализация экрана
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Defender Agent")

            # Загрузка изображений
            self.load_assets()
        elif self.screen is None:  # Если pygame инициализирован, но экран отсутствует
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Defender Agent")
            self.load_assets()

    def load_assets(self):
        assets = [
            ('ship_img', 'assets/ship.png', (SHIP_WIDTH, SHIP_HEIGHT)),
            ('meteor_img', 'assets/meteor.png', (60, 60)),
            ('human_img', 'assets/human.png', (HUMAN_SIZE, HUMAN_SIZE)),
            ('background_img', 'assets/background.png', (SCREEN_WIDTH, SCREEN_HEIGHT)),
            ('bullet_img', 'assets/bullet.png', (BULLET_WIDTH, BULLET_HEIGHT))
        ]

        for attr, path, size in assets:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Asset not found: {path}")
            img = pygame.image.load(path)
            setattr(self, attr, pygame.transform.scale(img, size))

    def reset(self, training=False):
        # Начальное состояние игры
        self.ship_x = SCREEN_WIDTH // 2
        self.ship_y = SCREEN_HEIGHT - 100

        # Сброс скоростей
        self.ship_velocity_x = 0
        self.ship_velocity_y = 0

        # Устанавливаем режим обучения
        self.training_mode = training

        # Восстановление здоровья при сбросе
        self.health = SHIP_HEALTH if not self.infinite_health else 0
        self.last_health = self.health  # Для отслеживания урона

        # Сброс параметров волны
        self.wave_count = 0
        self.meteors_in_wave = 0
        self.max_meteors_per_wave = WAVE_INITIAL_METEORS
        self.meteor_wave_timer = pygame.time.get_ticks()  # Сбрасываем таймер волны
        self.last_meteor_spawn = 0
        self.last_human_spawn = 0

        # Сброс счетчиков
        self.score = 0
        self.total_score = 0
        self.saved_humans = 0
        self.destroyed_meteors = 0
        self.last_saved_humans = []  # Сброс списка спасенных людей
        self.last_destroyed_meteors = []  # Сброс списка уничтоженных метеоритов
        self.shots_fired = 0
        self.shots_hit = 0
        self.damage_taken = 0
        self.last_meteor_details = []  # Добавляем инициализацию списка

        # В режиме обучения всегда одна жизнь
        if training:
            self.lives = 1
        else:
            if not hasattr(self, 'lives'):
                self.lives = 3

        # Инициализация остальных атрибутов при первом запуске
        if not hasattr(self, 'total_score'):
            self.total_score = 0
        if not hasattr(self, 'total_saved_humans'):
            self.total_saved_humans = 0
        if not hasattr(self, 'destroyed_meteors'):
            self.destroyed_meteors = 0
        if not hasattr(self, 'game_over'):
            self.game_over = False

        # Сброс информации о наградах
        self.reward_info = {
            'saved': 0,
            'destroyed': 0,
            'near_human': 0,
            'near_meteor': 0,
            'meteor_dodge': 0,
            'edge_penalty': 0,
            'accuracy_penalty': 0,
            'damage_penalty': 0,
            'accuracy': 0,  # Награда за точность
            'protection': 0,  # Награда за защиту людей
        }

        # Сброс атрибутов для отслеживания наград
        self.last_position = (self.ship_x, self.ship_y)
        self.inactivity_counter = 0
        self.last_meteor_details = []

        self.meteors = []
        self.humans = []
        self.bullets = []

        return self.get_state()

    def step(self, action, visualize=False, training=False):
        if visualize:
            self.clock.tick(FPS)

        if visualize:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.screen = None  # Сбрасываем экран
                        return None, 0, True

                self.render()
                pygame.display.flip()
            except pygame.error:
                print("Ошибка отрисовки pygame. Пересоздаем окно...")
                self.screen = None
                pygame.quit()
                pygame.init()
                self.init_pygame()

        # В режиме обучения пропускаем обработку game_over
        if not training and self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, 0, True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.reset(training=False)
                    return self.get_state(), 0, False

        # Обработка движения с учетом диагонали
        dx = 0
        dy = 0

        # Определяем направление движения
        if action in [MOVE_LEFT, MOVE_UP_LEFT, MOVE_DOWN_LEFT]:
            dx = -1
        elif action in [MOVE_RIGHT, MOVE_UP_RIGHT, MOVE_DOWN_RIGHT]:
            dx = 1

        if action in [MOVE_UP, MOVE_UP_LEFT, MOVE_UP_RIGHT]:
            dy = -1
        elif action in [MOVE_DOWN, MOVE_DOWN_LEFT, MOVE_DOWN_RIGHT]:
            dy = 1

        # Нормализация вектора движения для диагонального движения
        if dx != 0 and dy != 0:
            magnitude = (dx * dx + dy * dy) ** 0.5
            dx = dx / magnitude
            dy = dy / magnitude

        # Применяем ускорение с учетом направления
        self.ship_velocity_x += dx * SHIP_ACCELERATION
        self.ship_velocity_y += dy * SHIP_ACCELERATION

        # Применяем трение только если нет активного ускорения в данном направлении
        if dx == 0:
            self.ship_velocity_x *= SHIP_FRICTION
        if dy == 0:
            self.ship_velocity_y *= SHIP_FRICTION

        # Обнуляем очень маленькие скорости для предотвращения микродвижений
        if abs(self.ship_velocity_x) < SHIP_MIN_SPEED:
            self.ship_velocity_x = 0
        if abs(self.ship_velocity_y) < SHIP_MIN_SPEED:
            self.ship_velocity_y = 0

        # Ограничение максимальной скорости
        current_speed = (self.ship_velocity_x ** 2 + self.ship_velocity_y ** 2) ** 0.5
        if current_speed > SHIP_MAX_SPEED:
            speed_scale = SHIP_MAX_SPEED / current_speed
            self.ship_velocity_x *= speed_scale
            self.ship_velocity_y *= speed_scale

        # Обновление позиции с учетом delta_time
        self.ship_x += self.ship_velocity_x
        self.ship_y += self.ship_velocity_y

        # Ограничение движения в пределах экрана с плавным замедлением
        if self.ship_x < 0:
            self.ship_x = 0
            self.ship_velocity_x *= 0.5  # Плавное замедление при ударе о стенку
        elif self.ship_x > SCREEN_WIDTH - SHIP_WIDTH:
            self.ship_x = SCREEN_WIDTH - SHIP_WIDTH
            self.ship_velocity_x *= 0.5

        if self.ship_y < 0:
            self.ship_y = 0
            self.ship_velocity_y *= 0.5
        elif self.ship_y > SCREEN_HEIGHT - SHIP_HEIGHT:
            self.ship_y = SCREEN_HEIGHT - SHIP_HEIGHT
            self.ship_velocity_y *= 0.5

        # Стрельба
        if action == SHOOT and self.check_meteor_in_front():
            self.shoot()

        # Обновление игрового состояния
        self.update_game_state()

        # Вычисление счета
        score = self.calculate_score(training)

        if self.saved_humans > 0:
            self.total_saved_humans += self.saved_humans

        # Проверка завершения эпизода
        if training:
            done = self.health <= 0
            if done:
                self.reset(training=True)
        else:
            if self.health <= 0 and not self.game_over:
                self.lives -= 1
                if self.lives > 0:
                    self.health = SHIP_HEALTH
                    self.ship_x = SCREEN_WIDTH // 2
                    self.ship_y = SCREEN_HEIGHT - 100
                    self.ship_velocity_x = 0
                    self.ship_velocity_y = 0
                else:
                    self.game_over = True
            done = self.game_over

        return self.get_state(), score, done

    def update_score(self):
        # Базовый счет только за уничтожение метеоритов и спасение людей
        score = 0

        # Подсчет очков за уничтоженные метеориты
        if hasattr(self, 'last_destroyed_meteors'):
            for meteor_size in self.last_destroyed_meteors:
                self.reward_info['destroyed'] += REWARD_DESTROY_METEOR[meteor_size]
                score += REWARD_DESTROY_METEOR[meteor_size]

        # Подсчет очков за спасенных людей
        if hasattr(self, 'last_saved_humans'):
            self.reward_info['saved'] += REWARD_SAVE_HUMAN * len(self.last_saved_humans)
            score += REWARD_SAVE_HUMAN * len(self.last_saved_humans)

        return score

    def calculate_score(self, training=True):
        score = 0

        score += self.update_score()

        # Дополнительные награды только для обучения
        if training:
            # Награда за близость к людям
            for human in self.humans:
                distance = ((self.ship_x - human['x']) ** 2 +
                            (self.ship_y - human['y']) ** 2) ** 0.5
                # Награда только если человек находится выше корабля
                if distance < SCREEN_WIDTH * 0.15 and human['y'] < self.ship_y:
                    self.reward_info['near_human'] += REWARD_NEAR_HUMAN
                    score += REWARD_NEAR_HUMAN

            # Штрафы и награды за близость к метеоритам
            for meteor in self.meteors:
                distance = ((self.ship_x - meteor['x']) ** 2 +
                            (self.ship_y - meteor['y']) ** 2) ** 0.5

                # Штраф за опасную близость к метеориту (только если метеорит выше корабля)
                if distance < meteor['size'] * 2 and meteor['y'] < self.ship_y:
                    self.reward_info['near_meteor'] += REWARD_NEAR_METEOR
                    score += REWARD_NEAR_METEOR
                # Награда за успешное уклонение
                elif (distance < meteor['size'] * 4 and meteor['y'] < self.ship_y):
                    # Проверяем, что корабль движется в сторону от метеорита
                    meteor_vector_x = meteor['x'] - self.ship_x
                    # Если метеорит слева и корабль движется вправо или наоборот
                    if (meteor_vector_x > 0 and self.ship_velocity_x < -0.5) or \
                            (meteor_vector_x < 0 and self.ship_velocity_x > 0.5):
                        self.reward_info['meteor_dodge'] += REWARD_METEOR_DODGE
                        score += REWARD_METEOR_DODGE

            # Штраф за нахождение у края экрана
            edge_distance_x = min(self.ship_x, SCREEN_WIDTH - (self.ship_x + SHIP_WIDTH))
            edge_distance_y = min(self.ship_y, SCREEN_HEIGHT - (self.ship_y + SHIP_HEIGHT))

            # Определяем зоны штрафа (10% от размера экрана)
            danger_zone_x = SCREEN_WIDTH * 0.1
            danger_zone_y = SCREEN_HEIGHT * 0.1

            # Рассчитываем штраф в зависимости от близости к краю
            if edge_distance_x < danger_zone_x or edge_distance_y < danger_zone_y:
                # Плавный штраф, зависящий от близости к краю
                x_penalty = 0
                y_penalty = 0

                if edge_distance_x < danger_zone_x:
                    x_penalty = (1 - edge_distance_x / danger_zone_x) * 0.5

                if edge_distance_y < danger_zone_y:
                    # Для нижнего края экрана штраф меньше
                    if self.ship_y > SCREEN_HEIGHT / 2:
                        y_penalty = (1 - edge_distance_y / danger_zone_y) * 0.1  # Уменьшенный штраф внизу
                    else:
                        y_penalty = (1 - edge_distance_y / danger_zone_y) * 0.5  # Обычный штраф вверху

                # Общий штраф - максимум из штрафов по X и Y
                edge_penalty = REWARD_EDGE_PENALTY * max(x_penalty, y_penalty)
                self.reward_info['edge_penalty'] += edge_penalty
                score += edge_penalty

                # Небольшой дополнительный штраф при движении к краю на опасно близком расстоянии
                if edge_distance_x < SHIP_WIDTH * 1.5:
                    if ((self.ship_velocity_x < 0 and self.ship_x < SCREEN_WIDTH / 2) or
                            (self.ship_velocity_x > 0 and self.ship_x > SCREEN_WIDTH / 2)):
                        score += REWARD_EDGE_PENALTY * 0.15

                if edge_distance_y < SHIP_HEIGHT * 1.5:
                    # Уменьшенный штраф за движение к нижнему краю
                    movement_penalty = 0.2 if self.ship_y < SCREEN_HEIGHT / 2 else 0.05
                    if ((self.ship_velocity_y < 0 and self.ship_y < SCREEN_HEIGHT / 2) or
                            (self.ship_velocity_y > 0 and self.ship_y > SCREEN_HEIGHT / 2)):
                        score += REWARD_EDGE_PENALTY * movement_penalty

            # Штраф за промахи (только если были выстрелы)
            if self.shots_fired > 0:
                accuracy = self.shots_hit / self.shots_fired
                if accuracy < 0.5:  # Штраф только при низкой точности
                    # Количество промахов * штраф за промах
                    missed_shots = self.shots_fired - self.shots_hit
                    accuracy_penalty = missed_shots * REWARD_MISSED_SHOT
                    self.reward_info['accuracy_penalty'] += accuracy_penalty
                    score += accuracy_penalty
                # Сбрасываем счетчики после начисления штрафа
                self.shots_fired = 0
                self.shots_hit = 0

            # Штраф за полученный урон
            if self.damage_taken > 0:
                # Процент потерянного здоровья * коэффициент штрафа
                health_loss_percent = self.damage_taken / SHIP_HEALTH
                damage_penalty = health_loss_percent * REWARD_DAMAGE_PENALTY
                self.reward_info['damage_penalty'] += damage_penalty
                score += damage_penalty
                self.damage_taken = 0  # Сбрасываем накопленный урон после начисления штрафа

            # Награды за точность попадания и защиту людей
            if hasattr(self, 'last_meteor_details') and len(self.last_meteor_details) > 0:
                for meteor_info in self.last_meteor_details:
                    hit_accuracy = float(meteor_info.get('hit_accuracy', 0))  # Преобразуем в float для безопасности

                    # Награды за точность
                    if hit_accuracy >= 0.9:  # Идеальное попадание (90-100%)
                        self.reward_info['accuracy'] += REWARD_PERFECT_SHOT['perfect']
                        score += REWARD_PERFECT_SHOT['perfect']
                    elif hit_accuracy >= 0.6:  # Хорошее попадание (60-89%)
                        self.reward_info['accuracy'] += REWARD_PERFECT_SHOT['good']
                        score += REWARD_PERFECT_SHOT['good']
                    elif hit_accuracy >= 0.4:  # Нормальное попадание (40-59%)
                        self.reward_info['accuracy'] += REWARD_PERFECT_SHOT['normal']
                        score += REWARD_PERFECT_SHOT['normal']

                    # Проверяем, был ли метеорит близко к людям
                    closest_human_dist = meteor_info.get('closest_human_dist', float('inf'))
                    screen_diagonal = (SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5

                    if closest_human_dist < screen_diagonal * 0.1:  # Очень близко
                        self.reward_info['protection'] += REWARD_EFFICIENT_KILL['critical']
                        score += REWARD_EFFICIENT_KILL['critical']
                    elif closest_human_dist < screen_diagonal * 0.2:  # Близко
                        self.reward_info['protection'] += REWARD_EFFICIENT_KILL['close']
                        score += REWARD_EFFICIENT_KILL['close']
                    elif closest_human_dist < screen_diagonal * 0.3:  # Средне
                        self.reward_info['protection'] += REWARD_EFFICIENT_KILL['moderate']
                        score += REWARD_EFFICIENT_KILL['moderate']
                    else:  # Далеко
                        self.reward_info['protection'] += REWARD_EFFICIENT_KILL['far']
                        score += REWARD_EFFICIENT_KILL['far']
                # Очищаем список после обработки всех метеоритов
                self.last_meteor_details = []

        # Обновляем общий счет
        self.total_score += score

        return score

    def get_state(self):
        # Нормализация позиции корабля
        ship_x_norm = self.ship_x / SCREEN_WIDTH
        ship_y_norm = self.ship_y / SCREEN_HEIGHT

        # Нормализация здоровья
        health_norm = self.health / SHIP_HEALTH if not self.infinite_health else 1.0

        # Инициализация значений по умолчанию
        nearest_meteor_dist = 1.0
        nearest_meteor_size = 0.0
        nearest_human_dist = 1.0

        # Поиск ближайшего метеора
        if self.meteors:
            distances = [((m['x'] - self.ship_x) ** 2 + (m['y'] - self.ship_y) ** 2) ** 0.5
                         for m in self.meteors]
            min_dist = min(distances)
            nearest_meteor_dist = min(1.0, min_dist / ((SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5))
            nearest_meteor_size = self.meteors[distances.index(min_dist)]['size'] / max(METEOR_SIZES)

        # Поиск ближайшего человека
        if self.humans:
            distances = [((h['x'] - self.ship_x) ** 2 + (h['y'] - self.ship_y) ** 2) ** 0.5
                         for h in self.humans]
            nearest_human_dist = min(1.0, min(distances) / ((SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5))

        # Нормализация остальных параметров
        bullets_norm = min(1.0, len(self.bullets) / 10)  # Максимум 10 пуль
        score_norm = min(1.0, self.score / 1000)  # Нормализация счета

        state = np.array([
            ship_x_norm,  # [0] Нормализованная позиция корабля по X
            ship_y_norm,  # [1] Нормализованная позиция корабля по Y
            health_norm,  # [2] Нормализованное здоровье
            nearest_meteor_dist,  # [3] Расстояние до ближайшего метеора
            nearest_meteor_size,  # [4] Размер ближайшего метеора
            nearest_human_dist,  # [5] Расстояние до ближайшего человека
            bullets_norm,  # [6] Нормализованное количество пуль
            score_norm  # [7] Нормализованный счет
        ], dtype=np.float32)

        return state

    def update_game_state(self):
        current_time = pygame.time.get_ticks()
        self.delta_time = (current_time - self.last_update_time) / 1000.0  # конвертируем в секунды
        self.last_update_time = current_time

        # Обновляем состояние с учетом delta_time
        self.spawn_objects()
        self.update_objects()
        self.check_collisions()

    def spawn_objects(self):
        current_time = pygame.time.get_ticks()

        # Система волн метеоритов
        if current_time - self.meteor_wave_timer > WAVE_DURATION:
            self.wave_count += 1
            self.meteor_wave_timer = current_time
            self.meteors_in_wave = 0

            # Более плавное увеличение количества метеоритов
            wave_increase = min(self.wave_count // 2, WAVE_MAX_METEORS - WAVE_INITIAL_METEORS)
            self.max_meteors_per_wave = WAVE_INITIAL_METEORS + wave_increase

        # Increase meteor spawn rate and speed based on wave count and score
        if (current_time - self.last_meteor_spawn > METEOR_SPAWN_COOLDOWN and
                self.meteors_in_wave < self.max_meteors_per_wave and
                len(self.meteors) < METEOR_MAX_COUNT):

            # Adjust spawn chance and speed based on wave and score
            spawn_chance = 0.7 + min(0.2, self.total_score / 5000) + min(0.1, self.wave_count / 10)
            if random.random() < spawn_chance:
                size = random.choice(METEOR_SIZES)
                spawn_x = random.randint(0, SCREEN_WIDTH - size)

                # Ensure valid spawn position
                valid_position = True
                for meteor in self.meteors:
                    dist = ((spawn_x - meteor['x']) ** 2 + (size / 2) ** 2) ** 0.5
                    if dist < MIN_SPAWN_DISTANCE:
                        valid_position = False
                        break

                for human in self.humans:
                    dist = ((spawn_x - human['x']) ** 2) ** 0.5
                    if dist < MIN_SPAWN_DISTANCE:
                        valid_position = False
                        break

                if valid_position:
                    base_speed = random.uniform(*METEOR_SPEED_RANGE)
                    wave_speed_bonus = min(0.5, self.wave_count * WAVE_SPEED_INCREASE)
                    speed = base_speed * (1 + wave_speed_bonus)

                    meteor = {
                        'x': spawn_x,
                        'y': -size,
                        'size': size,
                        'speed': speed,
                        'health': METEOR_HEALTH[size]
                    }
                    self.meteors.append(meteor)
                    self.meteors_in_wave += 1
                    self.last_meteor_spawn = current_time
                    self.total_meteors_appeared += 1

        # Спавн людей
        if (current_time - self.last_human_spawn > HUMAN_SPAWN_COOLDOWN and
                len(self.humans) < HUMAN_MAX_COUNT):

            # Адаптивный спавн людей
            base_spawn_chance = HUMAN_SPAWN_RATE
            # Увеличиваем шанс спавна если людей мало
            if len(self.humans) < HUMAN_MAX_COUNT // 2:
                base_spawn_chance *= 1.5
            # Уменьшаем шанс если много метеоритов
            if len(self.meteors) > METEOR_MAX_COUNT // 2:
                base_spawn_chance *= 0.7

            if random.random() < base_spawn_chance:
                # Разделим экран на секции для более равномерного распределения
                section_width = SCREEN_WIDTH // SCREEN_SECTIONS

                # Выбираем секцию, где меньше всего объектов
                section_counts = [0] * SCREEN_SECTIONS
                for human in self.humans:
                    section = int(human['x'] // section_width)
                    section_counts[section] += 1
                for meteor in self.meteors:
                    section = int(meteor['x'] // section_width)
                    section_counts[section] += 1

                # Выбираем секцию с минимальным количеством объектов
                best_section = section_counts.index(min(section_counts))

                # Спавним в выбранной секции
                spawn_x = random.randint(
                    best_section * section_width,
                    (best_section + 1) * section_width - HUMAN_SIZE
                )

                human = {
                    'x': spawn_x,
                    'y': -HUMAN_SIZE,
                    'speed': random.uniform(*HUMAN_SPEED_RANGE)
                }
                self.humans.append(human)
                self.last_human_spawn = current_time
                self.total_humans_appeared += 1

    def update_objects(self):
        # Обновление позиций пуль
        for bullet in self.bullets[:]:  # Используем копию списка
            bullet['y'] -= BULLET_SPEED
            # Удаляем пули, вышедшие за пределы экрана
            if bullet['y'] < 0:
                self.bullets.remove(bullet)

        # Обновление позиций метеоритов
        for meteor in self.meteors:
            meteor['y'] += meteor['speed']
            # Если метеорит достиг низа экрана
            if meteor['y'] > SCREEN_HEIGHT:
                meteor['y'] = -meteor['size']
                meteor['x'] = random.randint(0, SCREEN_WIDTH - meteor['size'])

        # Обновление позиций людей
        for human in self.humans[:]:  # Используем копию списка
            human['y'] += human['speed']
            # Если человек достиг низа экрана, считаем его потерянным
            if human['y'] > SCREEN_HEIGHT:
                self.humans.remove(human)

    def check_meteor_in_front(self):
        # Определяем область перед кораблем
        ship_center_x = self.ship_x + SHIP_WIDTH // 2
        ship_top = self.ship_y

        # Увеличенная зона обнаружения метеоритов
        detection_zone_width = SHIP_WIDTH * 8  # Увеличиваем ширину зоны
        detection_zone_height = SCREEN_HEIGHT * 0.9  # Увеличиваем высоту зоны до всего экрана

        for meteor in self.meteors:
            # Предсказание будущей позиции метеорита
            future_meteor_y = meteor['y'] + meteor['speed'] * 5  # Предсказываем на 5 шагов вперед

            # Проверка попадания метеорита в зону обнаружения
            if (ship_center_x - detection_zone_width // 2 <= meteor[
                'x'] <= ship_center_x + detection_zone_width // 2 and
                    ship_top < future_meteor_y < ship_top + detection_zone_height):
                return True

        return False

    def shoot(self):
        current_time = pygame.time.get_ticks()

        if current_time - self.last_shot_time >= self.shooting_cooldown:
            self.bullets.append({
                'x': self.ship_x + SHIP_WIDTH // 2 - BULLET_WIDTH // 2,
                'y': self.ship_y
            })
            self.last_shot_time = current_time
            self.shots_fired += 1  # Увеличиваем счетчик выстрелов

    def check_collisions(self):
        # Проверка столкновений корабля с метеоритами
        ship_rect = pygame.Rect(self.ship_x, self.ship_y, SHIP_WIDTH, SHIP_HEIGHT)

        # Проверка столкновений с метеоритами
        self.last_destroyed_meteors = []  # Сброс списка уничтоженных метеоритов
        self.last_saved_humans = []  # Сброс списка спасенных людей

        # Проверка подбора людей
        for human in self.humans[:]:  # Используем копию списка
            human_rect = pygame.Rect(human['x'], human['y'], HUMAN_SIZE, HUMAN_SIZE)

            # Проверка столкновения человека с метеоритом
            for meteor in self.meteors[:]:
                meteor_rect = pygame.Rect(meteor['x'], meteor['y'], meteor['size'], meteor['size'])
                if human_rect.colliderect(meteor_rect):
                    if human in self.humans:  # Проверяем, что человек еще существует
                        self.humans.remove(human)
                    break

            if ship_rect.colliderect(human_rect):
                if human in self.humans:  # Проверяем, что человек еще существует
                    self.saved_humans += 1
                    self.total_saved_humans += 1
                    self.last_saved_humans.append('saved')  # Добавляем в список спасенных
                    self.humans.remove(human)

        # Проверка столкновений корабля с метеоритами
        for meteor in self.meteors[:]:  # Используем копию списка
            meteor_rect = pygame.Rect(meteor['x'], meteor['y'], meteor['size'], meteor['size'])

            # Проверка столкновения корабля с метеоритом
            if ship_rect.colliderect(meteor_rect):
                damage = meteor['size'] * 2  # Урон пропорционален размеру метеорита
                self.health -= damage
                self.meteors.remove(meteor)
                self.last_destroyed_meteors.append(meteor)

                # Если здоровье падает до нуля
                if self.health <= 0 and not self.training_mode:
                    self.game_over = True
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True

        # Проверка столкновений пуль с метеоритами
        for bullet in self.bullets[:]:
            bullet_rect = pygame.Rect(bullet['x'], bullet['y'], BULLET_WIDTH, BULLET_HEIGHT)
            hit = False

            for meteor in self.meteors[:]:
                meteor_rect = pygame.Rect(meteor['x'], meteor['y'], meteor['size'], meteor['size'])
                if bullet_rect.colliderect(meteor_rect):
                    meteor['health'] -= 1
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    self.shots_hit += 1  # Увеличиваем счетчик попаданий

                    # Вычисляем точность попадания
                    bullet_center = (bullet['x'] + BULLET_WIDTH / 2, bullet['y'] + BULLET_HEIGHT / 2)
                    meteor_center = (meteor['x'] + meteor['size'] / 2, meteor['y'] + meteor['size'] / 2)
                    distance = ((bullet_center[0] - meteor_center[0]) ** 2 +
                                (bullet_center[1] - meteor_center[1]) ** 2) ** 0.5

                    # Используем более мягкую формулу для точности
                    # Теперь даже попадание по краю даст точность около 0.4
                    normalized_distance = distance / meteor['size']  # Делим на полный размер, а не на радиус
                    current_accuracy = max(0.4,
                                           1.0 - normalized_distance)  # Минимальная точность 0.4 за любое попадание

                    hit = True
                    if meteor['health'] <= 0:
                        if meteor in self.meteors:
                            # Сохраняем информацию о метеорите перед удалением
                            # Найдем ближайшего человека в момент уничтожения метеорита
                            closest_human_dist = float('inf')
                            for human in self.humans:
                                dist = ((meteor['x'] - human['x']) ** 2 +
                                        (meteor['y'] - human['y']) ** 2) ** 0.5
                                closest_human_dist = min(closest_human_dist, dist)

                            # Сохраняем последнюю точность попадания для этого метеорита
                            meteor_info = {
                                'size': meteor['size'],
                                'x': meteor['x'],
                                'y': meteor['y'],
                                'hit_accuracy': current_accuracy,  # Используем текущую точность попадания
                                'closest_human_dist': closest_human_dist
                            }
                            self.meteors.remove(meteor)
                            # Добавляем только размер в список уничтоженных
                            self.last_destroyed_meteors.append(meteor['size'])
                            # Сохраняем детали метеорита
                            self.last_meteor_details.append(meteor_info)
                            self.destroyed_meteors += 1  # Увеличиваем общий счетчик
                    break

            # Проверка выхода пули за пределы экрана
            if not hit and bullet['y'] < 0 and bullet in self.bullets:
                self.bullets.remove(bullet)

        # Проверка столкновений корабля с метеоритами
        current_health = self.health
        for meteor in self.meteors[:]:
            meteor_rect = pygame.Rect(meteor['x'], meteor['y'], meteor['size'], meteor['size'])
            if ship_rect.colliderect(meteor_rect):
                if not self.infinite_health:
                    # Используем урон в зависимости от размера метеорита
                    damage = METEOR_DAMAGE[meteor['size']]
                    self.health -= damage
                if meteor in self.meteors:
                    self.meteors.remove(meteor)

        # Подсчет полученного урона
        damage = current_health - self.health
        if damage > 0:
            self.damage_taken += damage

    def render(self):
        if self.screen is None:
            self.init_pygame()  # Инициализируем pygame если экран не создан
            if self.screen is None:  # Если все еще None после инициализации
                return

        # Отрисовка фона
        self.screen.blit(self.background_img, (0, 0))

        # Отрисовка игровых объектов
        self.screen.blit(self.ship_img, (self.ship_x, self.ship_y))

        # Кэш для масштабированных изображений метеоритов
        for meteor in self.meteors:
            size = meteor['size']
            if size not in self.scaled_meteor_images:
                self.scaled_meteor_images[size] = pygame.transform.scale(self.meteor_img, (size, size))
            meteor['scaled_img'] = self.scaled_meteor_images[size]
            self.screen.blit(meteor['scaled_img'], (meteor['x'], meteor['y']))

        for human in self.humans:
            self.screen.blit(self.human_img, (human['x'], human['y']))

        for bullet in self.bullets:
            self.screen.blit(self.bullet_img, (bullet['x'], bullet['y']))

        # Отрисовка HUD
        font = pygame.font.Font(None, 36)
        # Отображение здоровья и жизней в зависимости от режима бесконечного здоровья
        if self.infinite_health:
            health_text = font.render('Здоровье: inf', True, WHITE)
            lives_text = font.render('Жизни: inf', True, WHITE)
        else:
            health_text = font.render(f'Здоровье: {max(0, int(self.health))}', True, WHITE)
            lives_text = font.render(f'Жизни: {self.lives}', True, WHITE)

        score_text = font.render(f"Счет: {self.total_score:.0f}", True, WHITE)
        humans_text = font.render(f"Спасено: {self.saved_humans}", True, WHITE)
        meteors_text = font.render(f"Уничтожено: {self.destroyed_meteors}", True, WHITE)

        # Кэш для текстов HUD
        hud_texts = [
            ('Здоровье: inf', (10, 10)) if self.infinite_health else (f'Здоровье: {max(0, int(self.health))}', (10, 10)),
            ('Жизни: inf', (10, 90)) if self.infinite_health else (f'Жизни: {self.lives}', (10, 90)),
            (f"Счет: {self.total_score:.0f}", (10, 50)),
            (f"Спасено: {self.saved_humans}", (10, 130)),
            (f"Уничтожено: {self.destroyed_meteors}", (10, 170))
        ]
        for text, pos in hud_texts:
            if text not in self.cached_texts:
                self.cached_texts[text] = font.render(text, True, WHITE)
            self.screen.blit(self.cached_texts[text], pos)

        # Отрисовка окна "Игра окончена"
        if self.game_over and (hasattr(self, 'training_mode') and not self.training_mode):
            # Создаем полупрозрачный фон
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            s.set_alpha(128)
            s.fill((0, 0, 0))
            self.screen.blit(s, (0, 0))

            # Отрисовка текста
            font_big = pygame.font.Font(None, 74)
            font = pygame.font.Font(None, 48)

            game_over_text = font_big.render('Игра окончена!', True, WHITE)
            final_score_text = font.render(f'Счет: {self.total_score}', True, WHITE)
            saved_humans_text = font.render(f'Спасено людей: {self.saved_humans}', True, WHITE)
            destroyed_meteors_text = font.render(f'Уничтожено метеоритов: {self.destroyed_meteors}', True, WHITE)

            text_y = SCREEN_HEIGHT // 2 - 100
            self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, text_y))
            self.screen.blit(final_score_text, (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, text_y + 80))
            self.screen.blit(saved_humans_text, (SCREEN_WIDTH // 2 - saved_humans_text.get_width() // 2, text_y + 140))
            self.screen.blit(destroyed_meteors_text,
                             (SCREEN_WIDTH // 2 - destroyed_meteors_text.get_width() // 2, text_y + 200))

        # Отрисовка информации о наградах (только во время обучения)
        if hasattr(self, 'reward_info') and hasattr(self, 'training_mode') and self.training_mode:
            # Шрифт для отображения наград
            font = pygame.font.Font(None, 24)
            y_offset = 10
            x_position = SCREEN_WIDTH - 300

            # Заголовок
            text = font.render("Награды и штрафы:", True, WHITE)
            self.screen.blit(text, (x_position, y_offset))
            y_offset += 25

            # Словарь с русскими названиями наград
            reward_names = {
                'destroyed': 'Уничтожение метеоритов',
                'saved': 'Спасение людей',
                'near_human': 'Близость к людям',
                'near_meteor': 'Близость к метеоритам',
                'meteor_dodge': 'Уклонение',
                'edge_penalty': 'Близко к краю',
                'accuracy_penalty': 'Плохая точность',
                'accuracy': 'Хорошая точность',
                'damage_penalty': 'Получение урона',
                'protection': 'Защита людей',
            }

            # Отображение каждой награды
            for key, value in self.reward_info.items():
                if value != 0:  # Показываем только ненулевые значения
                    color = GREEN if value > 0 else RED
                    text = font.render(f"{reward_names[key]}: {value:.2f}", True, color)
                    self.screen.blit(text, (x_position, y_offset))
                    y_offset += 20

        pygame.display.flip()  # Обновляем экран
