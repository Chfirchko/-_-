import os
from PyQt6.QtWidgets import (QMainWindow, QGraphicsView, QGraphicsScene,
                             QVBoxLayout, QWidget, QPushButton, QLabel,
                             QSpinBox, QFileDialog, QApplication)
from PyQt6.QtGui import QPainter, QBrush, QColor
from PyQt6.QtCore import Qt, QTimer
from config import *
from drone import Drone, Cluster
from environment import Environment
from simulation import Simulation

class SimulationView(QGraphicsView):
    def __init__(self, env, cluster):
        super().__init__()
        self.env = env
        self.cluster = cluster
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.cell_size = 30

    def draw_environment(self):
        self.scene.clear()

        # Отрисовка сетки
        for x in range(self.env.size):
            for y in range(self.env.size):
                if self.env.grid[x][y] == 1:
                    self.scene.addRect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                        brush=QBrush(Qt.GlobalColor.darkGray)
                    )

        # Отрисовка стартовой зоны (зеленая рамка)
        x0, y0, w, h = self.env.start_area
        self.scene.addRect(
            x0 * self.cell_size,
            y0 * self.cell_size,
            w * self.cell_size,
            h * self.cell_size,
            pen=Qt.GlobalColor.green
        )

        # Отрисовка финишной зоны (красная рамка)
        x0, y0, w, h = self.env.finish_area
        self.scene.addRect(
            x0 * self.cell_size,
            y0 * self.cell_size,
            w * self.cell_size,
            h * self.cell_size,
            pen=Qt.GlobalColor.red
        )

        # Отрисовка дронов
        for i, drone in enumerate(self.cluster.drones):
            if hasattr(drone, 'alive') and drone.alive:
                # Синий круг для дрона
                ellipse = self.scene.addEllipse(
                    drone.x * self.cell_size,
                    drone.y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                    brush=QBrush(Qt.GlobalColor.blue)
                )
                # Текст с номером
                text = self.scene.addText(str(i))
                text.setPos(
                    drone.x * self.cell_size + 5,
                    drone.y * self.cell_size + 5
                )
                text.setDefaultTextColor(Qt.GlobalColor.white)

        # Центр кластера (желтая точка)
        if self.cluster.center:
            cx, cy = self.cluster.center
            self.scene.addEllipse(
                cx * self.cell_size - 3,
                cy * self.cell_size - 3,
                6, 6,
                brush=QBrush(Qt.GlobalColor.yellow)
            )

        self.setSceneRect(0, 0,
                          self.env.size * self.cell_size,
                          self.env.size * self.cell_size)


class MainWindow(QMainWindow):
    def __init__(self, env, drones, cluster, simulation):
        super().__init__()
        self.env = env
        self.drones = drones
        self.cluster = cluster
        self.simulation = simulation
        self.drone_count = len(drones)

        self.init_ui()
        self.check_saved_model()

    def init_ui(self):
        self.setWindowTitle("Drone Swarm Learning")
        self.setGeometry(100, 100, 800, 900)

        container = QWidget()
        layout = QVBoxLayout()

        # Панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        self.size_spin = QSpinBox()
        self.size_spin.setRange(10, 50)
        self.size_spin.setValue(DEFAULT_GRID_SIZE)
        self.size_spin.valueChanged.connect(self.update_grid_size)

        self.drones_spin = QSpinBox()
        self.drones_spin.setRange(1, 50)
        self.drones_spin.setValue(DEFAULT_DRONE_COUNT)
        self.drones_spin.valueChanged.connect(self.update_drone_count)

        self.spawn_spin = QSpinBox()
        self.spawn_spin.setRange(1, 5)
        self.spawn_spin.setValue(3)
        self.spawn_spin.valueChanged.connect(self.update_spawn_size)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)

        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self.save_model)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)

        self.reset_btn = QPushButton("Reset Simulation")
        self.reset_btn.clicked.connect(self.reset_simulation)

        self.status_label = QLabel("Ready to start training")
        self.episode_label = QLabel(f"Episode: 0/{TRAINING_EPISODES}")
        self.alive_label = QLabel("Alive: 0/0")
        self.reward_label = QLabel("Average Reward: 0")
        self.time_label = QLabel("Time: 0s")

        control_layout.addWidget(QLabel("Grid Size:"))
        control_layout.addWidget(self.size_spin)
        control_layout.addWidget(QLabel("Drone Count:"))
        control_layout.addWidget(self.drones_spin)
        control_layout.addWidget(QLabel("Spawn Area Size:"))
        control_layout.addWidget(self.spawn_spin)
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.episode_label)
        control_layout.addWidget(self.alive_label)
        control_layout.addWidget(self.reward_label)
        control_layout.addWidget(self.time_label)

        control_panel.setLayout(control_layout)

        # Вид симуляции
        self.view = SimulationView(self.env, self.cluster)
        self.view.draw_environment()

        layout.addWidget(control_panel)
        layout.addWidget(self.view)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_status(self):
        alive_count = sum(1 for d in self.drones if d.alive)
        avg_reward = sum(d.total_reward for d in self.drones) / max(1, len(self.drones))
        elapsed = self.simulation.episode_timer.elapsed() / 1000

        self.episode_label.setText(f"Episode: {self.simulation.episode}/{TRAINING_EPISODES}")
        self.alive_label.setText(f"Alive: {alive_count}/{len(self.drones)}")
        self.reward_label.setText(f"Average Reward: {avg_reward:.2f}")
        self.time_label.setText(f"Time: {elapsed:.2f}s")

        # Перерисовываем сцену
        self.view.draw_environment()

    def start_training(self):
        if self.simulation.is_training:
            return

        self.simulation.is_training = True
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")

        # Принудительная отрисовка перед стартом
        self.view.draw_environment()
        QApplication.processEvents()

        # Запуск симуляции
        self.simulation.start_training(self.update_status)

    def stop_training(self):
        self.simulation.is_training = False
        self.simulation.timer.stop()
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Training stopped at episode {self.simulation.episode}")

    def reset_simulation(self):
        self.simulation.is_training = False
        self.simulation.episode = 0
        self.env = Environment(self.env.size, self.env.spawn_size)
        self.drones = [Drone() for _ in range(self.drone_count)]

        # Установка позиций дронов
        spawn_positions = self.env.get_valid_spawn_positions(len(self.drones))
        for i, drone in enumerate(self.drones):
            drone.x, drone.y = spawn_positions[i]

        self.cluster = Cluster(self.drones)
        self.simulation = Simulation(self.env, self.drones, self.cluster)
        self.view.env = self.env
        self.view.cluster = self.cluster
        self.view.draw_environment()
        self.update_status()

    def save_model(self):
        if not self.drones:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Pickle files (*.pkl)")
        if path:
            self.drones[0].save(path)
            self.status_label.setText(f"Model saved to {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle files (*.pkl)")
        if path:
            try:
                for drone in self.drones:
                    drone.load(path)
                self.status_label.setText(f"Model loaded from {path}")
            except Exception as e:
                self.status_label.setText(f"Error loading model: {str(e)}")

    def update_grid_size(self, size):
        self.env.size = size
        self.reset_simulation()

    def update_drone_count(self, count):
        self.drone_count = count
        self.reset_simulation()

    def update_spawn_size(self, size):
        self.env.spawn_size = size
        self.reset_simulation()

    def check_saved_model(self):
        if os.path.exists(MODEL_FILE):
            self.status_label.setText("Found saved model. Click 'Load Model' to load it.")
