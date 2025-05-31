import sys
from PyQt6.QtWidgets import QApplication
from config import *
from environment import Environment
from drone import Drone, Cluster
from simulation import Simulation
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Инициализация объектов
    env = Environment(DEFAULT_GRID_SIZE, spawn_size=3)
    drones = [Drone() for _ in range(DEFAULT_DRONE_COUNT)]

    # Установка позиций дронов
    spawn_positions = env.get_valid_spawn_positions(len(drones))
    for i, drone in enumerate(drones):
        drone.x, drone.y = spawn_positions[i]

    cluster = Cluster(drones)
    simulation = Simulation(env, drones, cluster)

    # Создание и отображение GUI
    window = MainWindow(env, drones, cluster, simulation)
    window.show()

    sys.exit(app.exec())