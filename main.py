import random

grid_size = 25

robot_x = 0
robot_y = 0

num_obstacles = 30
num_hyacinth = 15

obstacles = []
hyacinth = []


# generate obstacles
while len(obstacles) < num_obstacles:
    x = random.randint(0, grid_size - 1)
    y = random.randint(0, grid_size - 1)

    if (x, y) != (robot_x, robot_y):
        obstacles.append((x, y))


# generate hyacinth
while len(hyacinth) < num_hyacinth:
    x = random.randint(0, grid_size - 1)
    y = random.randint(0, grid_size - 1)

    if (x, y) != (robot_x, robot_y) and (x, y) not in obstacles:
        hyacinth.append((x, y))


def print_grid():

    for i in range(grid_size):
        row = []

        for j in range(grid_size):

            if (i, j) == (robot_x, robot_y):
                row.append("R")

            elif (i, j) in obstacles:
               row.append("○")

            elif (i, j) in hyacinth:
                row.append("H")

            else:
                row.append(".")

        print(" ".join(row))

    print()


def move_robot(action):
    global robot_x, robot_y

    new_x = robot_x
    new_y = robot_y

    if action == "UP" and robot_x > 0:
        new_x -= 1

    elif action == "DOWN" and robot_x < grid_size - 1:
        new_x += 1

    elif action == "LEFT" and robot_y > 0:
        new_y -= 1

    elif action == "RIGHT" and robot_y < grid_size - 1:
        new_y += 1

    if (new_x, new_y) in obstacles:
        print("Obstacle hit!")
        return

    robot_x = new_x
    robot_y = new_y

    if (robot_x, robot_y) in hyacinth:
        hyacinth.remove((robot_x, robot_y))
        print("Hyacinth removed!")


print("Initial Grid")
print_grid()

move_robot("RIGHT")
print_grid()

move_robot("DOWN")
print_grid()
