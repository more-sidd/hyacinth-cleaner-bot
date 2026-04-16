import random

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

#defining learning structure 
class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.actions = ACTIONS

    def get_q(self, state, action):
        #the agent back to already-visited (and now penalised) cells.
        return self.q_table.get((state, action), 0.1)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.get_q(state, a))

    #updating status
    def update(self, state, action, reward, next_state):
        max_q_next = max(self.get_q(next_state, a) for a in self.actions)
        old_q = self.get_q(state, action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)