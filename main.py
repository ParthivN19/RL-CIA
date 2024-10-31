from grid_environment import GridEnvironment
from agents import DP, QLearning, SARSA
import random

# Environment and parameters
GRID_SIZE = 100
START = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
GOAL = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
OBSTACLE_PERCENTAGE = 0.2
GAMMA = 0.9

def results(env, agents, episodes=100):
    results = {}
    for agent_name, agent in agents.items():
        total_steps = 0
        for _ in range(episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done:
                if isinstance(agent, (QLearning, SARSA)):
                    action_index = agent.choose_action(state)
                    action = ["up", "down", "left", "right"][action_index]
                else:
                    action = agent.choose_action(state)
                state, _, done = env.step(action)
                steps += 1
            total_steps += steps
        results[agent_name] = total_steps / episodes
    return results

# Initialize environment and agents
env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLE_PERCENTAGE)
agents = {
    "DP": DP(env, GAMMA),
    "Q-Learning": QLearning(env),
    "SARSA": SARSA(env)
}

# Run DP algorithm
agents["DP"].run_DP()

# Train Q-Learning and SARSA agents
agents["Q-Learning"].learn(episodes=1000)
agents["SARSA"].learn(episodes=1000)

# Benchmark results
best = results(env, agents)
print("Best Results:", best)
