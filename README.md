Clone the repository:
cd RL_CIA
_________________________________________

CIA-1 - K-Arm Bandit Recommendation System
This project implements a K-arm bandit algorithm using an epsilon-greedy strategy for a recommendation system in Python. It dynamically learns which items are most effective based on user feedback.

Overview
The K-arm bandit model balances exploration of new items with exploitation of known popular items to maximize user engagement.

________________________________________

CIA-2 - Grid Navigation
This project demonstrates Value Iteration, Q-Learning, and SARSA methods applied to a 100x100 grid with obstacles.
Each agent learns an optimal policy to navigate from a random start to a goal location, minimizing the steps taken.

Algorithms
Value Iteration: Uses dynamic programming for policy optimization.
Q-Learning: Off-policy RL method that learns the best action-values.
SARSA: On-policy RL method that learns based on the current policy.

Run the main.py file to initialize the environment, train agents and show results:
python main.py

Structure
grid_envt.py: Grid environment setup with obstacles.
agents.py: Agent classes for DP, Q-Learning, and SARSA.
main.py: Initializes environment, trains agents, and display results.
