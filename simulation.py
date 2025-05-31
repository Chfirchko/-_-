from config import *
import numpy as np
from PyQt6.QtCore import QTimer, QElapsedTimer
from environment import Environment  # Добавлен импорт Environment
from drone import Cluster
class Simulation:
    def __init__(self, env, drones, cluster):
        self.env = env
        self.drones = drones
        self.cluster = cluster
        self.timer = QTimer()
        self.episode_timer = QElapsedTimer()
        self.episode = 0
        self.is_training = False
        self.directions = [(0, -1), (1, -1), (1, 0), (1, 1),
                           (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    def start_training(self, update_callback):
        self.update_callback = update_callback
        self.is_training = True
        self.episode = 0
        self.reset_episode()
        self.timer.timeout.connect(self.training_step)
        self.timer.start(STEP_INTERVAL)

    def training_step(self):

        if not self.is_training or self.episode >= TRAINING_EPISODES:
            self.timer.stop()
            return

        # Обновляем всех дронов
        for drone in self.drones:
            if drone.alive:
                state = self.get_drone_state(drone)
                action = drone.act(state)
                self.move_drone(drone, action)

                # После действия — считаем награду за шаг
                reward = -STEP_PENALTY
                if drone.alive:
                    reward += SURVIVAL_REWARD
                    if self.cluster.center:
                        dx = drone.x - self.cluster.center[0]
                        dy = drone.y - self.cluster.center[1]
                        dist_to_center = np.sqrt(dx ** 2 + dy ** 2)
                        if dist_to_center < CLUSTER_RADIUS / 2:
                            reward += CLOSE_TO_CENTER_BONUS
                else:
                    reward = -100  # штраф за смерть

                drone.store_transition(state, action, reward)

        self.cluster.calculate_center()
        self.check_cluster_distance()

        # Проверяем завершение эпизода
        if self.check_episode_end():
            self.episode += 1
            if self.episode < TRAINING_EPISODES:
                self.reset_episode()
            else:
                self.is_training = False

        # Обновляем интерфейс
        self.update_callback()
        if self.episode % 10 == 0:
            avg_reward = np.mean([d.total_reward for d in self.drones])
            print(
                f"Эпизод {self.episode}, Средняя награда: {avg_reward:.2f}, Живых: {sum(d.alive for d in self.drones)}")

    def move_drone(self, drone, action):
        dx, dy = self.directions[action]
        new_x, new_y = drone.x + dx, drone.y + dy

        # Проверка выхода за границы и препятствий
        if (0 <= new_x < self.env.size and
                0 <= new_y < self.env.size and
                self.env.grid[new_x][new_y] == 0):

            drone.x, drone.y = new_x, new_y
            drone.steps += 1

            # Проверка достижения финиша
            if self.env.in_finish_area(new_x, new_y):
                drone.alive = False  # "Убираем" дрона
                drone.total_reward += 100  # Награда за достижение финиша

        else:
            drone.alive = False

    def check_cluster_distance(self):
        if self.cluster.center:
            for drone in self.drones:
                if drone.alive:
                    distance = np.sqrt((drone.x - self.cluster.center[0]) ** 2 +
                                       (drone.y - self.cluster.center[1]) ** 2)
                    if distance > CLUSTER_RADIUS:
                        drone.alive = False

    def check_episode_end(self):
        elapsed = self.episode_timer.elapsed() / 1000
        all_dead = all(not drone.alive for drone in self.drones)
        # Можно убрать проверку reached_finish, т.к. обработка теперь в move_drone
        # reached_finish = any(drone.x == self.env.finish[0] and drone.y == self.env.finish[1] for drone in self.drones)

        if elapsed > EPISODE_TIMEOUT or all_dead:
            # Рассчитываем расстояние от центра кластера до финиша
            finish_x, finish_y, fw, fh = self.env.finish_area
            if self.cluster.center:
                cx, cy = self.cluster.center
                distance_to_finish = np.sqrt((cx - finish_x) ** 2 + (cy - finish_y) ** 2)
            else:
                distance_to_finish = self.env.size * 2  # Максимально возможное расстояние

            distance_penalty = distance_to_finish * 0.5

            for drone in self.drones:
                # Награда за выживание
                if drone.alive:
                    drone.total_reward += 10

                    # Бонус за близость к центру кластера
                    if self.cluster.center:
                        distance_to_center = np.sqrt((drone.x - cx) ** 2 + (drone.y - cy) ** 2)
                        if distance_to_center < CLUSTER_RADIUS / 2:
                            drone.total_reward += 5

                drone.total_reward -= drone.steps * 0.1
                drone.total_reward -= distance_penalty

                drone.store_transition(np.zeros(12), 0, drone.total_reward)

            return True
        return False

    def reset_episode(self):
        self.episode_timer.restart()
        # Генерируем новую среду
        self.env = Environment(self.env.size, self.env.spawn_size)
        # Получаем новые позиции спавна
        spawn_positions = self.env.get_valid_spawn_positions(len(self.drones))
        # Обучаем всех дронов перед сбросом
        for drone in self.drones:
            drone.learn()


        for i, drone in enumerate(self.drones):
            drone.x, drone.y = spawn_positions[i]
            drone.alive = True
            drone.steps = 0
            drone.total_reward = 0
            drone.memory = []
        self.cluster = Cluster(self.drones)
        print(f"Эпизод сброшен. Дроны живы: {[d.alive for d in self.drones]}")

    def get_drone_state(self, drone):
        state = np.zeros(12)  # 8 + 2 (центр) + 2 (финиш)

        for i, (dx, dy) in enumerate(self.directions):
            dist = 0
            x, y = drone.x, drone.y
            while True:
                x += dx
                y += dy
                if (0 <= x < self.env.size and 0 <= y < self.env.size):
                    if self.env.grid[x][y] == 1:
                        break
                    dist += 1
                else:
                    break
            state[i] = dist / self.env.size

        # Центр кластера
        if self.cluster.center:
            cx, cy = self.cluster.center
            state[8] = (drone.x - cx) / self.env.size
            state[9] = (drone.y - cy) / self.env.size
        else:
            state[8], state[9] = 0, 0

        # Финиш (центр финишной области)
        fx, fy, fw, fh = self.env.finish_area
        fx_center = fx + fw // 2
        fy_center = fy + fh // 2

        state[10] = (drone.x - fx_center) / self.env.size
        state[11] = (drone.y - fy_center) / self.env.size

        return state

