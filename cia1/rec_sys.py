import numpy as np
import random

class KArmBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms  #Number of arms
        self.epsilon = epsilon  #Probability of exploration
        self.counts = np.zeros(n_arms)  #Number of times each arm is selected
        self.values = np.zeros(n_arms)  #Estimated rewrd for each arm

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        new_value = old_value + (1 / n) * (reward - old_value)
        self.values[chosen_arm] = new_value

n_arms = 5  # Assuming there are 5 movies to recommend
bandit = KArmBandit(n_arms=n_arms, epsilon=0.1)

rewards = [0.1, 0.2, 0.4, 0.5, 0.7]  #random assigned rewards for each movie

for _ in range(100):
    arm = bandit.select_arm()
    
    reward = rewards[arm] + np.random.normal(0, 0.1)  # adding some noise
    
    bandit.update(arm, reward)

print("Counts of each arm selected:", bandit.counts)
print("Estimated values of each arm:", bandit.values)
