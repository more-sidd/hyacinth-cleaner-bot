import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from qlearning import QLearningAgent

#parameters
NUM_EPISODES = 5000
MAX_STEPS    = 2000   

env   = Environment(size=25, num_obstacles=30, num_hyacinth=20)
agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon=1.0)


def get_state(env, vision_radius=9):
    rx, ry = env.robot

    #finding nearest target available 
    nearest = None
    min_dist = float('inf')
    for (hx, hy) in env.hyacinth:
        dist = abs(hx - rx) + abs(hy - ry)
        if dist <= vision_radius and dist < min_dist:
            min_dist = dist
            nearest = (hx, hy)

    #blind zone update (4x4 grid)
    blind_zone = None
    if nearest is None and env.hyacinth:
        zone_size = env.size // 4
        blind_zone = (rx // zone_size, ry // zone_size)

    return (env.robot, nearest, blind_zone)


#tracking updates
rewards_per_episode          = []
hyacinth_removed_per_episode = []
steps_per_episode            = []
completed_episodes           = []  

print("Training started...")
for episode in range(NUM_EPISODES):
    # Reshuffle map every 500 episodes to generalise,but keep stable between reshuffles
    reshuffle = (episode % 500 == 0)
    env.reset(reshuffle=reshuffle)
    state       = get_state(env)
    total_reward = 0
    initial_count = len(env.hyacinth)

    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        next_pos, step_reward, is_new = env.step(action)
        next_state = get_state(env)

        # Exploration bonus
        if next_state[1] is None and is_new:
            coverage = len(env.visited) / (env.size ** 2)
            step_reward += max(0.0, 2.0 * (1.0 - coverage))

        agent.update(state, action, step_reward, next_state)
        state = next_state
        total_reward += step_reward

        if env.is_done():
            break

    agent.decay_epsilon()

    removed = initial_count - len(env.hyacinth)
    rewards_per_episode.append(total_reward)
    hyacinth_removed_per_episode.append(removed)
    steps_per_episode.append(step + 1)
    completed_episodes.append(1 if env.is_done() else 0)

    if (episode + 1) % 100 == 0:
        window = completed_episodes[-100:]
        avg_rem  = sum(hyacinth_removed_per_episode[-100:]) / 100
        avg_rew  = sum(rewards_per_episode[-100:]) / 100
        comp_pct = sum(window) / len(window) * 100
        print(
            f"Ep {episode+1:5d}/{NUM_EPISODES} | "
            f"Avg Reward: {avg_rew:7.1f} | "
            f"Avg Removed: {avg_rem:.1f}/20 | "
            f"Completed: {comp_pct:.0f}% | "
            f"ε: {agent.epsilon:.3f}"
        )

print("Training complete!")


#learning curves subplots
window = 100
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

smoothed_steps   = np.convolve(steps_per_episode,            np.ones(window)/window, mode='valid')
smoothed_rewards = np.convolve(rewards_per_episode,          np.ones(window)/window, mode='valid')
smoothed_removed = np.convolve(hyacinth_removed_per_episode, np.ones(window)/window, mode='valid')
smoothed_comp    = np.convolve(completed_episodes,           np.ones(window)/window, mode='valid') * 100

axes[0].plot(smoothed_steps,   color='orange')
axes[0].set_title('Steps per Episode')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Steps')
axes[0].axhline(y=250, color='red', linestyle='--', alpha=0.5, label='Target')
axes[0].legend()

axes[1].plot(smoothed_rewards, color='blue')
axes[1].set_title('Total Reward per Episode')
axes[1].set_xlabel('Episode')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[2].plot(smoothed_removed, color='green',  label='Avg Removed')
axes[2].plot(smoothed_comp,    color='purple', label='% Completed', linestyle='--')
axes[2].set_title('Hyacinth Removed & Completion Rate')
axes[2].set_xlabel('Episode')
axes[2].set_ylim(0, 105)
axes[2].axhline(y=20,  color='green',  linestyle=':', alpha=0.5)
axes[2].axhline(y=100, color='purple', linestyle=':', alpha=0.5)
axes[2].legend()

plt.tight_layout()
plt.savefig('learning_results.png', dpi=150)


#visual plot
agent.epsilon = 0.0  
env.reset(reshuffle=False) 
state = get_state(env)
path  = [env.robot]
snapshot_obstacles = env.initial_obstacles.copy()
snapshot_hyacinth  = env.initial_hyacinth.copy()

for step in range(MAX_STEPS):
    action     = max(agent.actions, key=lambda a: agent.get_q(state, a))
    _, _, _    = env.step(action)
    next_state = get_state(env)
    state      = next_state
    path.append(env.robot)
    if env.is_done():
        break

removed_final = sum(1 for h in snapshot_hyacinth if h not in env.hyacinth)
print(f"\nGreedy rollout: {removed_final}/20 hyacinth removed in {len(path)} steps. "
      f"{'COMPLETE' if env.is_done() else 'incomplete'}")

grid_display = np.zeros((env.size, env.size))
for (x, y) in snapshot_obstacles: grid_display[x][y] = -1
for (x, y) in snapshot_hyacinth:  grid_display[x][y] =  2
for (x, y) in path:
    if grid_display[x][y] == 0:   grid_display[x][y] =  1
rx, ry = env.robot
grid_display[rx][ry] = 3

fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.colormaps.get_cmap('RdYlGn').resampled(5)
im = ax.imshow(grid_display, cmap=cmap, vmin=-1, vmax=3)
status = f"COMPLETE ({len(path)} steps)" if env.is_done() else f"Incomplete — {removed_final}/20 removed"
ax.set_title(f'Final Greedy Path — {status}')


from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkred',    label='Obstacle'),
    Patch(facecolor='lightgreen', label='Hyacinth (initial)'),
    Patch(facecolor='lightyellow',label='Path taken'),
    Patch(facecolor='darkgreen',  label='Final position'),
    Patch(facecolor='orange',     label='Unvisited'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('grid_visualization.png', dpi=150)

print("Saved: learning_results.png, grid_visualization.png")