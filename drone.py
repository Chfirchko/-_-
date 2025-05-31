import numpy as np
import pickle
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)



class Drone:
    def __init__(self, exploration_rate=0.05):
        # Нейросеть с добавлением случайности
        self.weights = [
            np.random.randn(12, 64) * 0.1,
            np.random.randn(64, 32) * 0.1,
            np.random.randn(32, 8) * 0.1
        ]
        self.biases = [
            np.zeros(64),
            np.zeros(32),
            np.zeros(8)
        ]

        self.learning_rate = 0.01
        self.gamma = 0.99

        self.memory = []  # (state, action, reward)
        self.replay_buffer = ReplayBuffer()

        self.x = 0
        self.y = 0
        self.alive = True
        self.steps = 0
        self.total_reward = 0
        self.exploration_rate = exploration_rate
        print(f"[DRONE] Создан новый дрон в ({self.x},{self.y}), alive={self.alive}")

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = self.relu(x @ self.weights[0] + self.biases[0])
        x = self.relu(x @ self.weights[1] + self.biases[1])
        return x @ self.weights[2] + self.biases[2]

    def store_transition(self, state, action, reward):
        if not isinstance(state, np.ndarray) or state.shape != (12,):
            print(
                f"[ERROR] Неверное состояние: {state} с shape={state.shape if isinstance(state, np.ndarray) else 'not array'}")
        self.replay_buffer.push((state, action, reward))
        self.memory.append((state, action, reward))
        self.total_reward += reward

    def learn(self, gamma=0.99, learning_rate=0.1):
        gamma = gamma if gamma is not None else self.gamma
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate

        if not self.memory:
            return

        # Вычислим возвраты (returns)
        G = 0
        returns = []
        for _, _, reward in reversed(self.memory):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Пакуем переходы для обучения батчем
        states = np.array([m[0] for m in self.memory])
        actions = np.array([m[1] for m in self.memory])

        for state, action, G in zip(states, actions, returns):
            # Forward pass
            z1 = state @ self.weights[0] + self.biases[0]
            a1 = self.relu(z1)
            z2 = a1 @ self.weights[1] + self.biases[1]
            a2 = self.relu(z2)
            z3 = a2 @ self.weights[2] + self.biases[2]
            probs = self._softmax(z3)

            # Градиент логарифма политики
            dlog = -probs
            dlog[action] += 1  # grad log pi

            grad_output = dlog * G  # масштабируем наградой

            # Градиенты для последнего слоя
            dw3 = np.outer(a2, grad_output)
            db3 = grad_output

            # Градиенты для второго слоя
            da2 = self.weights[2] @ grad_output
            dz2 = da2 * (z2 > 0)  # ReLU backprop
            dw2 = np.outer(a1, dz2)
            db2 = dz2

            # Градиенты для первого слоя
            da1 = self.weights[1] @ dz2
            dz1 = da1 * (z1 > 0)
            dw1 = np.outer(state, dz1)
            db1 = dz1

            # Обновление весов
            self.weights[2] += learning_rate * dw3
            self.biases[2] += learning_rate * db3
            self.weights[1] += learning_rate * dw2
            self.biases[1] += learning_rate * db2
            self.weights[0] += learning_rate * dw1
            self.biases[0] += learning_rate * db1

        self.memory = []

    def _softmax(self, x):
        x = x.astype(np.float64)  # повысим точность
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)  # защита от мусора
        e_x = np.exp(x - np.max(x))  # нормализация для стабильности
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0 or np.isnan(sum_e_x):
            return np.ones_like(x) / len(x)  # равномерное распределение
        return e_x / sum_e_x

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, 7)
        logits = self.forward(state)
        probs = self._softmax(logits)
        return np.random.choice(len(probs), p=probs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'exploration_rate': self.exploration_rate
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            self.exploration_rate = data.get('exploration_rate', 0.3)


class Cluster:
    def __init__(self, drones):
        self.drones = drones
        self.center = (0, 0)

    def calculate_center(self):
        alive = [d for d in self.drones if d.alive]
        if not alive:
            return None
        x = sum(d.x for d in alive) / len(alive)
        y = sum(d.y for d in alive) / len(alive)
        self.center = (x, y)
        return self.center