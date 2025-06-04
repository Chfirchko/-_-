import numpy as np
import random


class Environment:
    def __init__(self, size, spawn_size=3, seed=None):
        """Инициализация игрового поля.

        Args:
            size (int): Размер поля (size x size).
            spawn_size (int, optional): Размер стартовой и финишной зоны. По умолчанию 3.
            seed (int, optional): Сид для генератора случайных чисел. По умолчанию None.
        """
        self.size = size
        self.spawn_size = spawn_size
        self.grid = np.zeros((size, size))  # Создаем пустое поле (0 - пусто, 1 - препятствие)
        self.start_area = (0, 0, spawn_size, spawn_size)  # Стартовая зона (x, y, width, height)
        # Финишная зона того же размера, что и старт
        self.finish_area = (size - spawn_size, size - spawn_size, spawn_size, spawn_size)
        self.generate_obstacles(seed)  # Генерация препятствий

    def in_finish_area(self, x, y):
        """Проверяет, находится ли точка в финишной зоне.

        Args:
            x (int): Координата X.
            y (int): Координата Y.

        Returns:
            bool: True, если точка в финишной зоне.
        """
        x0, y0, w, h = self.finish_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def generate_obstacles(self, seed=None):
        """Генерирует препятствия на поле.

        Args:
            seed (int, optional): Сид для генератора случайных чисел. По умолчанию None.
        """
        if seed is not None:
            random.seed(seed)

        for x in range(self.size):
            for y in range(self.size):
                # Препятствия не генерируются в стартовой и финишной зонах
                if not self.in_spawn_area(x, y) and not self.in_finish_area(x, y):
                    if random.random() < 0.1:  # Вероятность генерации препятствия (Пока 0.0, можно изменить)
                        self.grid[x][y] = 1  # Устанавливаем препятствие

    def in_spawn_area(self, x, y):
        """Проверяет, находится ли точка в стартовой зоне.

        Args:
            x (int): Координата X.
            y (int): Координата Y.

        Returns:
            bool: True, если точка в стартовой зоне.
        """
        x0, y0, w, h = self.start_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def in_finish_area(self, x, y):
        """Проверяет, находится ли точка в финишной зоне.

        Args:
            x (int): Координата X.
            y (int): Координата Y.

        Returns:
            bool: True, если точка в финишной зоне.
        """
        x0, y0, w, h = self.finish_area
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def get_random_spawn_position(self):
        """Генерирует случайную позицию в стартовой зоне.

        Returns:
            tuple: Координаты (x, y) в стартовой зоне.
        """
        x0, y0, w, h = self.start_area
        x = random.randint(x0, x0 + w - 1)
        y = random.randint(y0, y0 + h - 1)
        return x, y

    def get_valid_spawn_positions(self, count):
        """Генерирует список валидных позиций для спавна.

        Args:
            count (int): Количество необходимых позиций.

        Returns:
            list: Список кортежей с координатами (x, y).
        """
        positions = []
        attempts = 0
        max_attempts = count * 10  # Максимальное количество попыток

        while len(positions) < count and attempts < max_attempts:
            x, y = self.get_random_spawn_position()
            if self.grid[x][y] == 0:  # Проверяем, что клетка свободна
                if (x, y) not in positions:  # Проверяем уникальность позиции
                    positions.append((x, y))
            attempts += 1

        if len(positions) < count:
            # Добавим позиции (0,0) если не хватило уникальных позиций
            for _ in range(count - len(positions)):
                positions.append((0, 0))

        return positions