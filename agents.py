import numpy as np
import random
from collections import defaultdict

class DP:
    def __init__(self, env, gamma, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros((env.grid_size, env.grid_size))

    def run_DP(self):
        actions = ["up", "down", "left", "right"]
        while True:
            delta = 0
            for x in range(self.env.grid_size):
                for y in range(self.env.grid_size):
                    if (x, y) == self.env.goal or self.env.grid[(x, y)] == -1:
                        continue
                    v = self.values[(x, y)]
                    max_value = float('-inf')
                    for action in actions:
                        self.env.state = (x, y)
                        new_state, reward, _ = self.env.step(action)
                        new_value = reward + self.gamma * self.values[new_state]
                        max_value = max(max_value, new_value)
                    self.values[(x, y)] = max_value
                    delta = max(delta, abs(v - self.values[(x, y)]))
            if delta < self.theta:
                break

class QLearning:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(["up", "down", "left", "right"])
        return np.argmax(self.q_table[state])

    def learn(self, episodes=1000):
        actions = ["up", "down", "left", "right"]
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(actions[action])
                best_next_action = np.argmax(self.q_table[new_state])
                td_target = reward + self.gamma * self.q_table[new_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_delta
                state = new_state

class SARSA:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(["up", "down", "left", "right"])
        return np.argmax(self.q_table[state])

    def learn(self, episodes=1000):
        actions = ["up", "down", "left", "right"]
        for _ in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(actions[action])
                next_action = self.choose_action(next_state)
                td_target = reward + self.gamma * self.q_table[next_state][next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_delta
                state, action = next_state, next_action
