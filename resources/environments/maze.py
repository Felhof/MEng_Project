import cv2
from gym import spaces
import numpy as np


class Maze:

    DIRECTIONS = 4
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    WHITE = (0, 0, 0)
    BLACK = (255, 255, 255)
    GREEN = (0, 255, 0)

    def __init__(self, size=10, start=(0, 0), goal=(9, 9), walls=None, render=False):
        self.size = size
        self.start = np.array(start)
        self.position = np.array(start)
        self.goal = np.array(goal)
        self.walls = walls

        self.action_space = spaces.Discrete(self.DIRECTIONS)
        self.observation_space = spaces.MultiDiscrete((self.size, self.size))

        self.render = render

    def step(self, action):
        if action == self.UP:
            move = np.array([0,1])
        elif action == self.RIGHT:
            move = np.array([1, 0])
        elif action == self.DOWN:
            move = np.array([0, -1])
        elif action == self.LEFT:
            move = np.array([-1, 0])

        next_position = self.position + move

        if next_position[0] >= self.size or next_position[1] >= self.size:
            next_position = self.position

        if next_position[0] <= 0 or next_position[1] <= 0:
            next_position = self.position

        for wall in self.walls:
            if wall[0, 0] <= next_position[0] < wall[1, 0] and wall[0, 1] <= next_position[1] < wall[1, 1]:
                next_position = self.position
                break

        self.position = next_position

        reward = np.linalg.norm(self.goal - self.position)

        observation = self.position
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.position = self.start
        observation = self.position
        return observation

    def render(self, mode='human'):
        if self.render:
            self.draw()

    def close(self):
        pass

    def draw(self, magnification=500, save=False):
        image = np.zeros([int(magnification), int(magnification), 3], dtype=np.uint8)
        bottom_left = (0, 0)
        top_right = (magnification * self.size, magnification * self.size)
        cv2.rectangle(image, bottom_left, top_right, self.WHITE, thickness=cv2.FILLED)
        cv2.rectangle(image, bottom_left, top_right, self.BLACK, thickness=int(magnification * 0.02))

        for wall in self.walls:
            wall_bottom_left = int(magnification * wall[0, 0])
            wall_top_right = int(magnification * wall[1, 1])
            cv2.rectangle(image, wall_bottom_left, wall_top_right, self.BLACK, thickness=cv2.FILLED)

        agent_centre = (int(self.position[0] * magnification), int(self.position * magnification))
        agent_radius = int(0.02 * magnification)
        agent_colour = self.BLACK
        cv2.circle(image, agent_centre, agent_radius, agent_colour, cv2.FILLED)

        goal_centre = (int(self.goal[0] * magnification), int(self.goal * magnification))
        goal_radius = int(0.02 * magnification)
        goal_colour = self.GREEN
        cv2.circle(image, goal_centre, goal_radius, goal_colour, cv2.FILLED)

        # Show the image
        cv2.imshow("Environment", image)
        if save:
            cv2.imwrite("Policy.png", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)

"""
    def draw_optimal_policy(self, agent, steps=5):
        state = agent.get_state()
        dqn = agent.get_dqn()

        for _ in range(steps):
            action = np.argmax(dqn.predict_q_value(state).detach().numpy())
            next_state = state + agent.discrete_action_to_continuous(action)
            line_start = (int(self.magnification * state[0]), int(self.magnification * (1 - state[1])))
            line_end = (int(self.magnification * next_state[0]), int(self.magnification * (1 - next_state[1])))
            cv2.line(self.image, line_start, line_end, (0, 255, 0), thickness=5)
            state = next_state
"""
