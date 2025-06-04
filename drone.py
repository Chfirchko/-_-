# Импорт необходимых библиотек
import numpy as np  # для работы с массивами и математических операций
import pickle  # для сохранения и загрузки объектов
import random  # для генерации случайных чисел

# Класс буфера воспроизведения для хранения переходов (опыт дрона)
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity  # максимальная емкость буфера
        self.memory = []  # список для хранения переходов

    # Метод для добавления перехода в буфер
    def push(self, transition):
        if len(self.memory) >= self.capacity:  # если буфер заполнен
            self.memory.pop(0)  # удаляем самый старый переход
        self.memory.append(transition)  # добавляем новый переход

    # Метод для случайной выборки батча переходов
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    # Метод для получения текущего размера буфера
    def __len__(self):
        return len(self.memory)


# Класс, представляющий дрона
class Drone:
    def __init__(self, exploration_rate=0.05):
        # Инициализация весов нейросети с небольшими случайными значениями
        self.weights = [
            np.random.randn(12, 64) * 0.1,  # веса первого слоя (12 входов, 64 нейрона)
            np.random.randn(64, 32) * 0.1,   # веса второго слоя (64 входа, 32 нейрона)
            np.random.randn(32, 8) * 0.1      # веса третьего слоя (32 входа, 8 выходов)
        ]
        # Инициализация смещений нулями
        self.biases = [
            np.zeros(64),  # смещения первого слоя
            np.zeros(32),  # смещения второго слоя
            np.zeros(8)    # смещения третьего слоя
        ]

        self.learning_rate = 0.01  # скорость обучения
        self.gamma = 0.99          # коэффициент дисконтирования

        self.memory = []  # память для хранения (состояние, действие, награда)
        self.replay_buffer = ReplayBuffer()  # буфер воспроизведения

        # Параметры дрона
        self.x = 0  # координата x
        self.y = 0  # координата y
        self.alive = True  # флаг "жив ли дрон"
        self.steps = 0  # количество сделанных шагов
        self.total_reward = 0  # общая награда
        self.exploration_rate = exploration_rate  # вероятность исследования (случайного действия)
        print(f"[DRONE] Создан новый дрон в ({self.x},{self.y}), alive={self.alive}")

    # Функция активации ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Прямой проход через нейросеть
    def forward(self, x):
        x = self.relu(x @ self.weights[0] + self.biases[0])  # первый слой
        x = self.relu(x @ self.weights[1] + self.biases[1])   # второй слой
        return x @ self.weights[2] + self.biases[2]           # третий слой (выход)

    # Метод для сохранения перехода (состояние, действие, награда)
    def store_transition(self, state, action, reward):
        if not isinstance(state, np.ndarray) or state.shape != (12,):
            print(
                f"[ERROR] Неверное состояние: {state} с shape={state.shape if isinstance(state, np.ndarray) else 'not array'}")
        self.replay_buffer.push((state, action, reward))  # сохраняем в буфер воспроизведения
        self.memory.append((state, action, reward))       # сохраняем в память
        self.total_reward += reward                       # увеличиваем общую награду

    # Метод для обучения на основе накопленного опыта
    def learn(self, gamma=0.99, learning_rate=0.1):
        gamma = gamma if gamma is not None else self.gamma
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate

        if not self.memory:  # если память пуста, нечего учить
            return

        # Вычисляем возвраты (returns) с учетом дисконтирования
        G = 0
        returns = []
        for _, _, reward in reversed(self.memory):  # идем с конца
            G = reward + gamma * G  # дисконтированная награда
            returns.insert(0, G)    # добавляем в начало списка

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # нормализация

        # Подготовка данных для обучения
        states = np.array([m[0] for m in self.memory])    # массив состояний
        actions = np.array([m[1] for m in self.memory])   # массив действий

        # Обучение для каждого перехода
        for state, action, G in zip(states, actions, returns):
            # Прямой проход
            z1 = state @ self.weights[0] + self.biases[0]  # первый слой
            a1 = self.relu(z1)                            # активация ReLU
            z2 = a1 @ self.weights[1] + self.biases[1]    # второй слой
            a2 = self.relu(z2)                            # активация ReLU
            z3 = a2 @ self.weights[2] + self.biases[2]    # третий слой
            probs = self._softmax(z3)                     # вероятности действий

            # Вычисление градиента логарифма политики
            dlog = -probs
            dlog[action] += 1  # grad log pi

            grad_output = dlog * G  # масштабирование наградой

            # Обратное распространение ошибки (backpropagation)
            # Градиенты для последнего слоя
            dw3 = np.outer(a2, grad_output)
            db3 = grad_output

            # Градиенты для второго слоя
            da2 = self.weights[2] @ grad_output
            dz2 = da2 * (z2 > 0)  # производная ReLU
            dw2 = np.outer(a1, dz2)
            db2 = dz2

            # Градиенты для первого слоя
            da1 = self.weights[1] @ dz2
            dz1 = da1 * (z1 > 0)
            dw1 = np.outer(state, dz1)
            db1 = dz1

            # Обновление весов с учетом learning rate
            self.weights[2] += learning_rate * dw3
            self.biases[2] += learning_rate * db3
            self.weights[1] += learning_rate * dw2
            self.biases[1] += learning_rate * db2
            self.weights[0] += learning_rate * dw1
            self.biases[0] += learning_rate * db1

        self.memory = []  # очищаем память после обучения

    # Функция softmax для преобразования выходов в вероятности
    def _softmax(self, x):
        x = x.astype(np.float64)  # повышение точности
        # Защита от NaN и бесконечностей
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        e_x = np.exp(x - np.max(x))  # нормализация для численной стабильности
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0 or np.isnan(sum_e_x):
            return np.ones_like(x) / len(x)  # равномерное распределение в случае ошибки
        return e_x / sum_e_x

    # Метод для выбора действия
    def act(self, state):
        if random.random() < self.exploration_rate:  # с вероятностью exploration_rate
            return random.randint(0, 7)              # выбираем случайное действие
        logits = self.forward(state)                # получаем выходы нейросети
        probs = self._softmax(logits)               # преобразуем в вероятности
        return np.random.choice(len(probs), p=probs)  # выбираем действие согласно вероятностям

    # Метод для сохранения модели
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,              # сохраняем веса
                'biases': self.biases,               # сохраняем смещения
                'exploration_rate': self.exploration_rate  # сохраняем вероятность исследования
            }, f)

    # Метод для загрузки модели
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']            # загружаем веса
            self.biases = data['biases']              # загружаем смещения
            self.exploration_rate = data.get('exploration_rate', 0.3)  # загружаем вероятность исследования


# Класс кластера дронов
class Cluster:
    def __init__(self, drones):
        self.drones = drones    # список дронов в кластере
        self.center = (0, 0)    # центр кластера

    # Метод для вычисления центра кластера
    def calculate_center(self):
        alive = [d for d in self.drones if d.alive]  # только живые дроны
        if not alive:
            return None
        # Вычисляем средние координаты
        x = sum(d.x for d in alive) / len(alive)
        y = sum(d.y for d in alive) / len(alive)
        self.center = (x, y)
        return self.center