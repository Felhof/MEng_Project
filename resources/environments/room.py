import cv2
import gym
from gym import spaces
import numpy as np
import torch


class Room(gym.Env):

    AGENT_SIZE = 0.3

    DIRECTIONS = 4
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)

    def __init__(self, size=10, area=None, observation_space=None, entrypoints=None, goal=None, walls=None,
                 lower_levels=None, episode_length=300, show=False):
        self.size = size
        self.area = area
        self.entrypoints = entrypoints
        self.position = entrypoints[np.random.choice(len(entrypoints))]
        self.goal = np.array(goal)
        self.walls = walls
        self.lower_lvls = lower_levels
        self.max_distance = np.sqrt(2*size**2)

        self.action_space = spaces.Discrete(self.DIRECTIONS)
        self.observation_space = observation_space

        self.current_step = 1
        self.episode_length = episode_length
        self.show = show

    def step(self, action):
        if action == self.UP:
            move = np.array([0, self.AGENT_SIZE])
        elif action == self.RIGHT:
            move = np.array([self.AGENT_SIZE, 0])
        elif action == self.DOWN:
            move = np.array([0, -self.AGENT_SIZE])
        elif action == self.LEFT:
            move = np.array([-self.AGENT_SIZE, 0])

        next_position = self.position + move

        if next_position[0] >= self.size or next_position[1] >= self.size:
            next_position = self.position

        if next_position[0] <= 0 or next_position[1] <= 0:
            next_position = self.position

        for wall in self.walls:
            if wall[0][0] <= next_position[0] < wall[1][0] and wall[0][1] <= next_position[1] < wall[1][1]:
                next_position = self.position
                break

        in_area = self.area.inside(next_position)

        done = False
        if not in_area:
            for (lower_lvl_area, lower_lvl_model) in self.lower_lvls:
                if lower_lvl_area.inside(next_position):
                    state_tensor = torch.tensor(next_position).unsqueeze(0)
                    _, value, _ = lower_lvl_model.policy.forward(state_tensor)
                    reward = value.item() * (1 - self.current_step / self.episode_length)
                    done = True
                    break
            next_position = self.position

        if not done:
            reward = self.max_distance - np.linalg.norm(self.goal - next_position)
            self.current_step += 1
            done = self.current_step > self.episode_length

        self.position = next_position
        observation = self.position
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.current_step = 1
        self.position = self.entrypoints[np.random.choice(len(self.entrypoints))]
        observation = self.position
        return observation

    def render(self, mode='human'):
        if self.show:
            self.draw()

    def close(self):
        pass

    def draw(self, magnification=50, save=False):
        image = np.zeros([int(magnification * self.size), int(magnification * self.size), 3], dtype=np.uint8)
        bottom_left = (0, 0)
        top_right = (magnification * self.size, magnification * self.size)
        cv2.rectangle(image, bottom_left, top_right, self.WHITE, thickness=cv2.FILLED)
        cv2.rectangle(image, bottom_left, top_right, self.BLACK, thickness=int(magnification * 0.02))

        for wall in self.walls:
            wall_bottom_left = (magnification * wall[0][0], magnification * wall[0][1])
            wall_top_right = (magnification * wall[1][0], magnification * wall[1][1])
            cv2.rectangle(image, wall_bottom_left, wall_top_right, self.BLACK, thickness=cv2.FILLED)

        agent_centre = (int(self.position[0] * magnification), int(self.position[1] * magnification))
        agent_radius = int(self.AGENT_SIZE * magnification)
        agent_colour = self.BLACK
        cv2.circle(image, agent_centre, agent_radius, agent_colour, cv2.FILLED)

        goal_centre = (int(self.goal[0] * magnification), int(self.goal[1] * magnification))
        goal_radius = int(self.AGENT_SIZE * magnification)
        goal_colour = self.GREEN
        cv2.circle(image, goal_centre, goal_radius, goal_colour, cv2.FILLED)

        # Show the image
        cv2.imshow("Environment", image)
        if save:
            cv2.imwrite("Policy.png", image)
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


class Area:

    def __init__(self, rectangles):
        self.rectangles = rectangles

    def inside(self, position):
        for rect in self.rectangles:
            if rect[0][0] <= position[0] <= rect[1][0] and rect[0][1] <= position[1] <= rect[1][1]:
                return True
        return False