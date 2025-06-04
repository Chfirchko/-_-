from config import *
import numpy as np
from PyQt6.QtCore import QTimer, QElapsedTimer
from environment import Environment
from drone import Cluster


class Simulation:
    """Класс, управляющий процессом обучения кластера дронов."""

    def __init__(self, env, drones, cluster):
        """
        Args:
            env (Environment): Объект окружения
            drones (list): Список дронов
            cluster (Cluster): Кластер для управления группой дронов
        """
        self.env = env
        self.drones = drones
        self.cluster = cluster
        self.timer = QTimer()  # Таймер для шагов симуляции
        self.episode_timer = QElapsedTimer()  # Таймер измерения длительности эпизода
        self.episode = 0  # Текущий номер эпизода
        self.is_training = False  # Флаг процесса обучения

        # 8 возможных направлений движения (по часовой стрелке)
        self.directions = [(0, -1), (1, -1), (1, 0), (1, 1),
                           (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    def start_training(self, update_callback):
        """Запуск процесса обучения.

        Args:
            update_callback (function): Функция для обновления интерфейса
        """
        self.update_callback = update_callback
        self.is_training = True
        self.episode = 0
        self.reset_episode()
        self.timer.timeout.connect(self.training_step)
        self.timer.start(STEP_INTERVAL)  # Интервал из config.py

    def training_step(self):
        """Один шаг обучения для всех дронов."""
        if not self.is_training or self.episode >= TRAINING_EPISODES:
            self.timer.stop()
            return

        # Обновляем состояние всех дронов
        for drone in self.drones:
            if drone.alive:
                # Получаем текущее состояние дрона
                state = self.get_drone_state(drone)

                # Получаем действие от политики дрона
                action = drone.act(state)

                # Двигаем дрон
                self.move_drone(drone, action)

                # Расчет награды
                reward = -STEP_PENALTY  # Штраф за каждый шаг
                if drone.alive:
                    reward += SURVIVAL_REWARD  # Награда за выживание

                    # Дополнительная награда за близость к центру кластера
                    if self.cluster.center:
                        dx = drone.x - self.cluster.center[0]
                        dy = drone.y - self.cluster.center[1]
                        dist_to_center = np.sqrt(dx ** 2 + dy ** 2)
                        if dist_to_center < CLUSTER_RADIUS / 2:
                            reward += CLOSE_TO_CENTER_BONUS
                else:
                    reward = -100  # Большой штраф за "смерть"

                # Сохраняем опыт для обучения с подкреплением
                drone.store_transition(state, action, reward)

        # Пересчитываем центр кластера
        self.cluster.calculate_center()

        # Проверяем расстояние до центра кластера
        self.check_cluster_distance()

        # Проверка условий завершения эпизода
        if self.check_episode_end():
            self.episode += 1
            if self.episode < TRAINING_EPISODES:
                self.reset_episode()
            else:
                self.is_training = False

        # Обновляем интерфейс и логируем прогресс
        self.update_callback()
        if self.episode % 10 == 0:
            avg_reward = np.mean([d.total_reward for d in self.drones])
            print(f"Эпизод {self.episode}, Средняя награда: {avg_reward:.2f}, "
                  f"Живых: {sum(d.alive for d in self.drones)}")

    def move_drone(self, drone, action):
        """Перемещает дрон в соответствии с выбранным действием.

        Args:
            drone (Drone): Объект дрона
            action (int): Индекс направления движения (0-7)
        """
        dx, dy = self.directions[action]
        new_x, new_y = drone.x + dx, drone.y + dy

        # Проверка допустимости нового положения
        if (0 <= new_x < self.env.size and
                0 <= new_y < self.env.size and
                self.env.grid[new_x][new_y] == 0):

            drone.x, drone.y = new_x, new_y
            drone.steps += 1

            # Проверка достижения финишной зоны
            if self.env.in_finish_area(new_x, new_y):
                drone.alive = False
                drone.total_reward += 100  # Большая награда за финиш
        else:
            drone.alive = False  # Дрон "погибает" при столкновении

    def check_cluster_distance(self):
        """Проверяет, не отстал ли дрон от кластера."""
        if self.cluster.center:
            for drone in self.drones:
                if drone.alive:
                    distance = np.sqrt((drone.x - self.cluster.center[0]) ** 2 +
                                       (drone.y - self.cluster.center[1]) ** 2)
                    if distance > CLUSTER_RADIUS:
                        drone.alive = False  # Дрон "погибает", если отстал

    def check_episode_end(self):
        """Проверяет условия завершения текущего эпизода."""
        elapsed = self.episode_timer.elapsed() / 1000  # Время в секундах
        all_dead = all(not drone.alive for drone in self.drones)

        # Эпизод завершается по таймауту или если все дроны "погибли"
        if elapsed > EPISODE_TIMEOUT or all_dead:
            # Рассчитываем штраф за расстояние до финиша
            finish_x, finish_y, fw, fh = self.env.finish_area
            if self.cluster.center:
                cx, cy = self.cluster.center
                distance_to_finish = np.sqrt((cx - finish_x) ** 2 + (cy - finish_y) ** 2)
            else:
                distance_to_finish = self.env.size * 2  # Максимальное расстояние

            distance_penalty = distance_to_finish * 0.5

            # Начисляем финальные награды
            for drone in self.drones:
                if drone.alive:
                    drone.total_reward += 10  # Награда за выживание

                    # Дополнительная награда за близость к центру
                    if self.cluster.center:
                        distance_to_center = np.sqrt((drone.x - cx) ** 2 + (drone.y - cy) ** 2)
                        if distance_to_center < CLUSTER_RADIUS / 2:
                            drone.total_reward += 5

                # Штрафы за количество шагов и расстояние до финиша
                drone.total_reward -= drone.steps * 0.1
                drone.total_reward -= distance_penalty

                # Сохраняем финальное состояние
                drone.store_transition(np.zeros(12), 0, drone.total_reward)

            return True
        return False

    def reset_episode(self):
        """Сбрасывает состояние для нового эпизода обучения."""
        self.episode_timer.restart()

        # Генерируем новое окружение
        self.env = Environment(self.env.size, self.env.spawn_size)

        # Получаем новые позиции для дронов
        spawn_positions = self.env.get_valid_spawn_positions(len(self.drones))

        # Обучаем дронов на накопленном опыте
        for drone in self.drones:
            drone.learn()

        # Сбрасываем состояние дронов
        for i, drone in enumerate(self.drones):
            drone.x, drone.y = spawn_positions[i]
            drone.alive = True
            drone.steps = 0
            drone.total_reward = 0
            drone.memory = []

        self.cluster = Cluster(self.drones)
        print(f"Эпизод сброшен. Дроны живы: {[d.alive for d in self.drones]}")

    def get_drone_state(self, drone):
        """Формирует вектор состояния для дрона.

        Args:
            drone (Drone): Объект дрона

        Returns:
            np.array: Вектор состояния (12 элементов)
        """
        state = np.zeros(12)  # [8 направлений + 2 центра + 2 финиша]

        # 1. Сканирование окружения в 8 направлениях
        for i, (dx, dy) in enumerate(self.directions):
            dist = 0
            x, y = drone.x, drone.y
            while True:
                x += dx
                y += dy
                if (0 <= x < self.env.size and 0 <= y < self.env.size):
                    if self.env.grid[x][y] == 1:  # Обнаружено препятствие
                        break
                    dist += 1
                else:  # Выход за границы
                    break
            state[i] = dist / self.env.size  # Нормализация

        # 2. Относительные координаты центра кластера
        if self.cluster.center:
            cx, cy = self.cluster.center
            state[8] = (drone.x - cx) / self.env.size
            state[9] = (drone.y - cy) / self.env.size
        else:
            state[8], state[9] = 0, 0

        # 3. Относительные координаты финиша
        fx, fy, fw, fh = self.env.finish_area
        fx_center = fx + fw // 2
        fy_center = fy + fh // 2

        state[10] = (drone.x - fx_center) / self.env.size
        state[11] = (drone.y - fy_center) / self.env.size

        return state