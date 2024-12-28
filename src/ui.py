import glob
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, BooleanVar
from .config import *


class GameUI:
    def __init__(self, game, agent):
        self.root = tk.Tk()
        self.root.title("Defender AI Обучение")
        self.root.geometry("400x600")

        self.game = game
        self.agent = agent

        self.setup_ui()

    def validate_positive_int(self, value, field_name, min_value=1, max_value=None):
        try:
            int_value = int(value)
            if int_value < min_value:
                raise ValueError(f"{field_name} должно быть не меньше {min_value}")
            if max_value and int_value > max_value:
                raise ValueError(f"{field_name} должно быть не больше {max_value}")
            return int_value
        except ValueError:
            raise ValueError(f"{field_name} должно быть целым числом")

    def validate_positive_float(self, value, field_name, min_value=0, max_value=None, allow_zero=False):
        try:
            float_value = float(value)

            # Проверка на минимальное значение
            if not allow_zero and float_value <= 0:
                raise ValueError(f"{field_name} должно быть положительным числом")

            if allow_zero and float_value < 0:
                raise ValueError(f"{field_name} не может быть отрицательным")

            if min_value is not None and float_value < min_value:
                raise ValueError(f"{field_name} должно быть не меньше {min_value}")

            if max_value is not None and float_value > max_value:
                raise ValueError(f"{field_name} должно быть не больше {max_value}")

            return float_value
        except ValueError:
            raise ValueError(f"{field_name} должно быть числом")

    def setup_ui(self):
        training_frame = ttk.LabelFrame(self.root, text="Параметры обучения")
        training_frame.pack(padx=10, pady=10, fill="x")

        training_params = [
            {
                "label": "Episodes:",
                "default": f"{EPISODES}",
                "validator": lambda x: self.validate_positive_int(x, "Количество эпизодов", min_value=10,
                                                                  max_value=1000000)
            },
            {
                "label": "Learning Rate:",
                "default": f"{LEARNING_RATE}",
                "validator": lambda x: self.validate_positive_float(x, "Learning Rate", min_value=1e-5, max_value=1.0)
            },
            {
                "label": "Gamma:",
                "default": f"{GAMMA}",
                "validator": lambda x: self.validate_positive_float(x, "Gamma", min_value=0.0, max_value=1.0)
            },
            {
                "label": "Epsilon Start:",
                "default": f"{EPSILON_START}",
                "validator": lambda x: self.validate_positive_float(x, "Epsilon Start", min_value=0.0, max_value=1.0)
            },
            {
                "label": "Epsilon Decay:",
                "default": f"{EPSILON_DECAY}",
                "validator": lambda x: self.validate_positive_float(x, "Epsilon Decay", min_value=0.9, max_value=0.9999)
            },
            {
                "label": "Epsilon Min:",
                "default": f"{EPSILON_MIN}",
                "validator": lambda x: self.validate_positive_float(x, "Epsilon Min", min_value=0.0, max_value=1.0)
            },
            {
                "label": "Batch Size:",
                "default": f"{BATCH_SIZE}",
                "validator": lambda x: self.validate_positive_int(x, "Batch Size", min_value=1, max_value=512)
            }
        ]

        # Создание сетки для выравнивания
        for i, param in enumerate(training_params):
            label = ttk.Label(training_frame, text=param["label"])
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)

            entry = ttk.Entry(training_frame, width=20)
            entry.insert(0, param["default"])
            entry.grid(row=i, column=1, padx=5, pady=2, sticky='ew')

            attr_name = param["label"].lower().replace(' ', '_').replace(':', '')
            setattr(self, f"{attr_name}_entry", entry)
            setattr(self, f"{attr_name}_validator", param["validator"])

        # Настройка расширения столбцов
        training_frame.grid_columnconfigure(1, weight=1)

        # Чекбокс для бесконечного здоровья
        self.infinite_health_var = BooleanVar(value=False)
        self.infinite_health_check = ttk.Checkbutton(
            training_frame,
            text="Бесконечное здоровье",
            variable=self.infinite_health_var
        )
        self.infinite_health_check.grid(row=len(training_params), column=0, columnspan=2, pady=5, sticky='w')

        # Чекбокс для визуализации обучения
        self.visualize_training_var = BooleanVar(value=False)
        self.visualize_training_check = ttk.Checkbutton(
            training_frame,
            text="Визуализировать обучение",
            variable=self.visualize_training_var
        )
        self.visualize_training_check.grid(row=len(training_params) + 1, column=0, columnspan=2, pady=5, sticky='w')

        # Кнопки управления
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        train_btn = ttk.Button(btn_frame, text="Обучение", command=self.start_training)
        train_btn.pack(side=tk.LEFT, padx=5)

        play_btn = ttk.Button(btn_frame, text="Играть", command=self.play_game)
        play_btn.pack(side=tk.LEFT, padx=5)

        restart_btn = ttk.Button(btn_frame, text="Заного", command=self.restart_game)
        restart_btn.pack(side=tk.LEFT, padx=5)

        # Область логов
        self.log_text = tk.Text(self.root, height=15, width=50)
        self.log_text.pack(padx=10, pady=10)

    def play_game(self):
        try:
            # Загрузка последней модели
            latest_model = "results/models/defender_dqn_last_episode"

            # Проверка существования файла модели
            if not os.path.exists(latest_model):
                models = glob.glob("results/models/defender_dqn_episode_*")
                if models:
                    latest_model = max(models, key=os.path.getctime)

            if not os.path.exists(latest_model):
                raise FileNotFoundError("Модель не найдена. Сначала обучите агента.")

            # Загрузка модели
            self.agent.load_model(path=latest_model)

            # Установка бесконечного здоровья
            self.game.infinite_health = self.infinite_health_var.get()

            # Запуск в отдельном потоке
            threading.Thread(target=self.play_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")

    def start_training(self):
        try:
            # Валидация всех параметров с использованием динамических валидаторов
            validation_params = [
                ("episodes", "Episodes"),
                ("learning_rate", "Learning Rate"),
                ("gamma", "Gamma"),
                ("epsilon_start", "Epsilon Start"),
                ("epsilon_decay", "Epsilon Decay"),
                ("epsilon_min", "Epsilon Min"),
                ("batch_size", "Batch Size")
            ]

            # Список для хранения провалидированных значений
            validated_params = {}

            # Валидация каждого параметра
            for attr, name in validation_params:
                # Получаем entry и validator
                entry = getattr(self, f"{attr}_entry")
                validator = getattr(self, f"{attr}_validator")

                # Валидация значения
                try:
                    validated_value = validator(entry.get())
                    validated_params[attr] = validated_value
                except ValueError as e:
                    # Подсвечиваем невалидное поле
                    entry.configure(foreground='red')
                    messagebox.showerror("Ошибка валидации", str(e))
                    return

            # Передача всех параметров в поток обучения
            threading.Thread(target=self.train_thread, kwargs={
                'steps': validated_params['episodes'],
                'learning_rate': validated_params['learning_rate'],
                'gamma': validated_params['gamma'],
                'epsilon_start': validated_params['epsilon_start'],
                'epsilon_decay': validated_params['epsilon_decay'],
                'epsilon_min': validated_params['epsilon_min'],
                'batch_size': validated_params['batch_size']
            }, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Непредвиденная ошибка: {e}")

    def train_thread(self, steps, learning_rate, gamma, epsilon_start, epsilon_decay, epsilon_min, batch_size):
        try:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Начало обучения DQN ({steps} эпизодов)...\n")

            # Обновляем параметры агента
            self.agent.learning_rate = learning_rate
            self.agent.gamma = gamma
            self.agent.epsilon = epsilon_start
            self.agent.epsilon_decay = epsilon_decay
            self.agent.epsilon_min = epsilon_min
            self.agent.batch_size = batch_size

            # Обучение DQN с опциональной визуализацией
            def progress_callback(current_episode):
                # Добавляем 1 к current_episode для отсчета с 1
                progress_percent = ((current_episode + 1) / steps) * 100
                progress_message = f"Обучение завершено на {int(progress_percent)}%\n"
                self.root.after(0, self.update_log, progress_message)

            self.agent.train(
                total_episodes=steps,
                visualize=self.visualize_training_var.get(),
                progress_callback=progress_callback
            )

            self.root.after(0, self.update_log, "Обучение завершено!\n")
        except Exception as e:
            self.root.after(0, self.update_log, f"Ошибка обучения: {e}\n")

    def update_log(self, message):
        # Потокобезопасный метод обновления лога
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)

    def play_thread(self):
        try:
            # Отладочная печать
            print(f"Игра начата с бесконечным здоровьем: {self.game.infinite_health}")

            # Очищаем лог
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Запуск игры с обученным агентом...\n")

            # Запускаем игру один раз с правильными параметрами
            self.agent.play(
                render=True,
                infinite_health=self.game.infinite_health
            )
        except Exception as e:
            error_msg = f"Ошибка игры: {e}\n"
            print(error_msg)  # Для отладки
            self.root.after(0, lambda: self.log_text.insert(tk.END, error_msg))

    def restart_game(self):
        # Логика перезапуска игры
        self.game.lives = 3
        self.game.game_over = False
        self.game.total_score = 0
        self.game.total_saved_humans = 0
        self.game.health = SHIP_HEALTH
        self.game.score = 0
        self.game.saved_humans = 0
        self.game.meteors = []
        self.game.humans = []
        self.game.bullets = []
        self.game.ship_x = SCREEN_WIDTH // 2
        self.game.ship_y = SCREEN_HEIGHT - 100
        self.log_text.delete(1.0, tk.END)
        self.update_log("Игра перезапущена.")

    def run(self):
        self.root.mainloop()
