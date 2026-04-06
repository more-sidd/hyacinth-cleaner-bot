import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from qlearning import QLearningAgent
 
#setting no. of eps
NUM_EPISODES = 1000
MAX_STEPS = 1000
 
env = Environment(size=25, num_obstacles=30, num_hyacinth=15)
agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.5)
 
rewards_per_episode = []
hyacinth_removed_per_episode = []
 
 
#goal state
def get_state(env):
    nearest = None
    min_dist = float('inf')
    rx, ry = env.robot
 
    for (hx, hy) in env.hyacinth:
        dist = abs(hx - rx) + abs(hy - ry)
        if dist < min_dist:
            min_dist = dist
            nearest = (hx, hy)
 
    return (env.robot, nearest)
 
 
#training loop
print("Training started...")
 
for episode in range(NUM_EPISODES):
    env.reset()
    state = get_state(env)
    total_reward = 0
    initial_hyacinth = len(env.hyacinth)
 
    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        _, reward = env.step(action)
        next_state = get_state(env)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
 
        if env.is_done():
            break
 
    agent.decay_epsilon()
    rewards_per_episode.append(total_reward)
    removed = initial_hyacinth - len(env.hyacinth)
    hyacinth_removed_per_episode.append(removed)
 
    if (episode + 1) % 50 == 0:
        avg_reward = sum(rewards_per_episode[-50:]) / 50
        avg_removed = sum(hyacinth_removed_per_episode[-50:]) / 50
        print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.1f} | Avg Hyacinth Removed: {avg_removed:.1f} | Epsilon: {agent.epsilon:.3f}")
 
print("Training complete!")
 
#plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
window = 20
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window) / window, mode='valid')
smoothed_hyacinth = np.convolve(hyacinth_removed_per_episode, np.ones(window) / window, mode='valid')
 
ax1.plot(smoothed_rewards, color='steelblue')
ax1.set_title('Total Reward per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.grid(True, alpha=0.3)
 
ax2.plot(smoothed_hyacinth, color='green')
ax2.set_title('Hyacinth Removed per Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Plants Removed')
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
print("Learning curves saved as learning_curves.png")
 
#Final grid 
env.reset()
state = get_state(env)
path = [env.robot]
 
for step in range(MAX_STEPS):
    action = max(agent.actions, key=lambda a: agent.get_q(state, a))
    _, _ = env.step(action)
    next_state = get_state(env)
    state = next_state
    path.append(env.robot)
 
    if env.is_done():
        break
 

grid_display = np.zeros((env.size, env.size))
 
for (x, y) in env.initial_obstacles:
    grid_display[x][y] = -1       # obstacles
 
for (x, y) in env.initial_hyacinth:
    grid_display[x][y] = 2        # hyacinth locations
 
for (x, y) in path:
    if grid_display[x][y] == 0:
        grid_display[x][y] = 1   # robot path
 
rx, ry = env.robot
grid_display[rx][ry] = 3          # final robot position
 
fig2, ax = plt.subplots(figsize=(8, 8))
cmap = plt.colormaps.get_cmap('RdYlGn').resampled(5)
ax.imshow(grid_display, cmap=cmap, vmin=-1, vmax=3)
ax.set_title('Final Grid: Robot Path After Training')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
 
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkred', label='Obstacle'),
    Patch(facecolor='lightyellow', label='Robot Path'),
    Patch(facecolor='lightgreen', label='Hyacinth'),
    Patch(facecolor='darkgreen', label='Robot (final pos)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
 
plt.tight_layout()
plt.savefig('grid_visualization.png', dpi=150)
print("Grid visualization saved as grid_visualization.png")
 