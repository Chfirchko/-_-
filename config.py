import os

# Основные параметры симуляции
DEFAULT_GRID_SIZE = 10 # стандартный размер сетки
DEFAULT_DRONE_COUNT = 2  # начальное-кол-во дронов
CLUSTER_RADIUS = 5  # радиус кластера
EPISODE_TIMEOUT = 0.1  # секунд на эпизод
TRAINING_EPISODES = 9999999999999999999  # кол-во итераций
MODEL_FILE = "drone_model.pkl"  # название модели
STEP_INTERVAL = 1  # интервал итераций в мс

# Награды и штрафы
FINISH_REWARD = 1000
SURVIVAL_REWARD = 100
CLOSE_TO_CENTER_BONUS = 50
STEP_PENALTY = 0.1
DISTANCE_PENALTY_FACTOR = 0.5
