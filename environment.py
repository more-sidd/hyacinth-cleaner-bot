import random

class Environment:
    #pond grid setup with robot, obstacles and hyacinth
    def __init__(self, size=25, num_obstacles=30, num_hyacinth=15):
        self.size = size
        self.num_obstacles = num_obstacles
        self.num_hyacinth = num_hyacinth
        self.robot = (0, 0)
        self.obstacles = []
        self.hyacinth = []
        self._place_objects()
        self.initial_obstacles = self.obstacles.copy()  
        self.initial_hyacinth = self.hyacinth.copy()   

    #drop obstacles and hyacinth randomly
    def _place_objects(self):
        self.obstacles = []
        self.hyacinth = []

        while len(self.obstacles) < self.num_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != self.robot and pos not in self.obstacles:
                self.obstacles.append(pos)

        while len(self.hyacinth) < self.num_hyacinth:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != self.robot and pos not in self.obstacles and pos not in self.hyacinth:
                self.hyacinth.append(pos)

    #when the ep is done, start clean for new ep
    def reset(self):
        self.robot = (0, 0)
        self.obstacles = self.initial_obstacles.copy() 
        self.hyacinth = self.initial_hyacinth.copy()   
        return self.robot

    #move the robot and return reward
    def step(self, action):
        x, y = self.robot
        reward = -1  # small penalty per step to encourage efficiency

        if action == "UP":
            x -= 1
        elif action == "DOWN":
            x += 1
        elif action == "LEFT":
            y -= 1
        elif action == "RIGHT":
            y += 1

        # boundary check
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return self.robot, -5  # penalty for hitting wall

        # obstacle check
        if (x, y) in self.obstacles:
            return self.robot, -10  # penalty for hitting obstacle

        self.robot = (x, y)

        # hyacinth check
        if self.robot in self.hyacinth:
            self.hyacinth.remove(self.robot)
            reward = 20  # reward for removing hyacinth

        return self.robot, reward

    #checking if all hyacinth is removed
    def is_done(self):
        return len(self.hyacinth) == 0

    def print_grid(self):
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.robot:
                    row.append("R")
                elif (i, j) in self.obstacles:
                    row.append("o")
                elif (i, j) in self.hyacinth:
                    row.append("H")
                else:
                    row.append(".")
            print(" ".join(row))
        print()
