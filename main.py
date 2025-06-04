import sys
from PyQt6.QtWidgets import QApplication
from config import *
from environment import Environment
from drone import Drone, Cluster
from simulation import Simulation
from ui import MainWindow

if __name__ == "__main__":
    # Инициализация Qt-приложения
    app = QApplication(sys.argv)

    # Создание окружения (сетки) с заданным размером и зоной спавна
    # DEFAULT_GRID_SIZE - размер сетки из config.py
    # spawn_size=3 - размер зоны, где могут появляться дроны
    env = Environment(DEFAULT_GRID_SIZE, spawn_size=3)

    # Создание списка дронов (количество берется из config.py)
    drones = [Drone() for _ in range(DEFAULT_DRONE_COUNT)]

    # Получение валидных позиций для спавна дронов (без коллизий)
    # Количество позиций = количеству дронов
    spawn_positions = env.get_valid_spawn_positions(len(drones))

    # Установка позиций для каждого дрона
    for i, drone in enumerate(drones):
        drone.x, drone.y = spawn_positions[i]  # Присваиваем координаты из полученных позиций

    # Создание кластера дронов (управляющей структуры)
    cluster = Cluster(drones)

    # Создание симуляции, связывающей окружение, дронов и кластер
    simulation = Simulation(env, drones, cluster)

    # Создание главного окна приложения
    # Передаем в него все основные компоненты системы
    window = MainWindow(env, drones, cluster, simulation)
    window.show()  # Показываем окно

    # Запуск основного цикла приложения Qt
    # sys.exit для корректного завершения
    sys.exit(app.exec())