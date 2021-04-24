import gym
import numpy as np
import torch


class RoomEnvironment(gym.Env):

    def __init__(self, maze=None, room=None,
                 lower_levels=None, episode_length=300, show=False):
        self.maze = maze
        self.maze.position = room.reset()
        self.room = room
        self.lower_lvls = lower_levels

        self.action_space = self.maze.action_space
        self.observation_space = self.maze.observation_space

        self.current_step = 1
        self.episode_length = episode_length
        self.show = show

    def step(self, action):
        next_position = self.maze.move(action)

        in_room = self.room.inside(next_position, self.maze.agent_radius)

        done = False
        if not in_room:
            for (lower_lvl_room, lower_lvl_model) in self.lower_lvls:
                if lower_lvl_room.inside(next_position):
                    state_tensor = torch.tensor(next_position.to_numpy()).unsqueeze(0)
                    _, value, _ = lower_lvl_model.policy.forward(state_tensor)
                    reward = value.item() * (1 - self.current_step / self.episode_length)
                    done = True
                    break
            if not done:
                next_position = self.maze.position

        if not done:
            reward = self.maze.size - np.linalg.norm(self.maze.goal.to_numpy() - next_position.to_numpy())
            self.current_step += 1
            done = self.current_step > self.episode_length

        self.maze.position = next_position
        observation = self.maze.position.to_numpy()
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.current_step = 1
        self.maze.position = self.room.reset()
        observation = self.maze.position
        return observation.to_numpy()

    def render(self, mode='human'):
        self.maze.draw()

    def close(self):
        pass
