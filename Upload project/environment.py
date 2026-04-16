import random

class Environment:
    def __init__(self, size=25, num_obstacles=30, num_hyacinth=20):
        self.size = size
        self.num_obstacles = num_obstacles
        self.num_hyacinth = num_hyacinth
        self.robot = (0, 0)
        self.obstacles = []
        self.hyacinth = []
        self.visited = set()
        self._place_objects()

    def _place_objects(self):
        self.obstacles = []
        self.hyacinth = []
        while len(self.obstacles) < self.num_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != (0, 0) and pos not in self.obstacles:
                self.obstacles.append(pos)
        while len(self.hyacinth) < self.num_hyacinth:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != (0, 0) and pos not in self.obstacles and pos not in self.hyacinth:
                self.hyacinth.append(pos)
        self.initial_obstacles = self.obstacles.copy()
        self.initial_hyacinth = self.hyacinth.copy()

    def reset(self, reshuffle=False):
        self.robot = (0, 0)
        if reshuffle or not self.obstacles:
            self._place_objects()
        else:
            self.hyacinth = self.initial_hyacinth.copy()
        self.visited = {self.robot}
        return self.robot

    def step(self, action):
        x, y = self.robot
        reward = -1  # small step cost

        if action == "UP":    x -= 1
        elif action == "DOWN":  x += 1
        elif action == "LEFT":  y -= 1
        elif action == "RIGHT": y += 1

        #pond boundary loop
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return self.robot, -5, False

        #obsctable loop
        if (x, y) in self.obstacles:
            return self.robot, -10, False

        self.robot = (x, y)
        is_new_cell = self.robot not in self.visited
        self.visited.add(self.robot)

        if self.robot in self.hyacinth:
            self.hyacinth.remove(self.robot)
            bonus = 100 if len(self.hyacinth) == 0 else 0
            reward = 20 + bonus
        elif not is_new_cell:
            #revisiting cell penalty
            reward = -2

        return self.robot, reward, is_new_cell

    def is_done(self):
        return len(self.hyacinth) == 0