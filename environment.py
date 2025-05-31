import numpy as np
import random


class Environment:
    def __init__(self, size, spawn_size=3, seed=None):
        self.size = size
        self.spawn_size = spawn_size
        self.grid = np.zeros((size, size))
        self.start_area = (0, 0, spawn_size, spawn_size)
        # Финишная зона того же размера, что и старт
        self.finish_area = (size - spawn_size, size - spawn_size, spawn_size, spawn_size)
        self.generate_obstacles(seed)

    def in_finish_area(self, x, y):
        x0, y0, w, h = self.finish_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def generate_obstacles(self, seed=None):
        if seed is not None:
            random.seed(seed)

        for x in range(self.size):
            for y in range(self.size):
                if not self.in_spawn_area(x, y) and not self.in_finish_area(x, y):
                    if random.random() < 0.1:  # (Пока 0.0, можно изменить)
                        self.grid[x][y] = 1

    def in_spawn_area(self, x, y):
        x0, y0, w, h = self.start_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def in_finish_area(self, x, y):
        x0, y0, w, h = self.finish_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def get_random_spawn_position(self):
        x0, y0, w, h = self.start_area
        x = random.randint(x0, x0 + w - 1)
        y = random.randint(y0, y0 + h - 1)
        return x, y

    def get_valid_spawn_positions(self, count):
        positions = []
        attempts = 0
        max_attempts = count * 10

        while len(positions) < count and attempts < max_attempts:
            x, y = self.get_random_spawn_position()
            if self.grid[x][y] == 0:
                if (x, y) not in positions:
                    positions.append((x, y))
            attempts += 1

        if len(positions) < count:
            # Добавим позиции (0,0) если не хватило
            for _ in range(count - len(positions)):
                positions.append((0, 0))

        return positions
