import numpy as np
import random

class GridEnvironment:
    def __init__(self, grid_size, start, goal, obstacle_percentage):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.grid = np.zeros((grid_size, grid_size))
        self.set_obstacles(obstacle_percentage)
        self.state = start

    def set_obstacles(self, percentage):
        num_obstacles = int(self.grid_size ** 2 * percentage)
        for _ in range(num_obstacles):
            obstacle = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if obstacle != self.start and obstacle != self.goal:
                self.grid[obstacle] = -1

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == "up" and x > 0: x -= 1
        elif action == "down" and x < self.grid_size - 1: x += 1
        elif action == "left" and y > 0: y -= 1
        elif action == "right" and y < self.grid_size - 1: y += 1
        new_state = (x, y)
        reward = -1
        if new_state == self.goal:
            reward = 100
        elif self.grid[new_state] == -1:
            reward = -100  # Obstacle penalty
        self.state = new_state
        return new_state, reward, new_state == self.goal