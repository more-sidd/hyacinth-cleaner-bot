import random

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class QLearningAgent:
    #defining speed, sight and exploration rates of the robot
    def __init__(self, alpha=0.1, gamma=0.9, epsilon = 0.5):
        self.alpha = alpha      # learning rate
        self.gamma = gamma      
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}
        self.actions = ACTIONS

    #feasability of a move in a position 
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    #exploring and picking best move wrt epsilon
    def choose_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.get_q(state, a))

    #apply qlearning
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        max_q_next = max([self.get_q(next_state, a) for a in self.actions])
        old_q = self.get_q(state, action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self, decay_rate=0.998, min_epsilon=0.1):  
        # gradually reducing exploration over time
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
