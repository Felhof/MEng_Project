import gym
import numpy as np
from gym import spaces


class ResourceAllocationEnvironmentBase(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, ra_problem, max_timesteps=500):
        super(ResourceAllocationEnvironmentBase, self).__init__()
        self.ra_problem = ra_problem
        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.max_resource_availabilities = self.ra_problem.get_max_resource_availabilities()

        self.action_space = None
        self.observation_dim = None
        self.observation_space = None

        resource_observation_dim = self.ra_problem.get_max_resource_availabilities() + 1
        new_task_observation_dim = np.ones(self.ra_problem.get_task_count()) + 1
        running_task_observation_dim = np.ones(self.ra_problem.get_task_count()) * (max(resource_observation_dim) + 1)

        self.build_action_and_observation_space(new_task_observation_dim, running_task_observation_dim)

    def build_action_and_observation_space(self, new_task_observation_dim,
                                           running_task_observation_dim):
        self.action_space = spaces.MultiBinary(self.ra_problem.get_task_count())
        self.observation_dim = np.append(new_task_observation_dim, running_task_observation_dim)
        self.observation_space = spaces.MultiDiscrete(
            self.observation_dim
        )

    def close(self):
        pass

    def enumerate_observations(self):
        observations = []
        observation = np.zeros(len(self.observation_dim))
        observations.append(observation)
        observation = np.copy(observation)

        idx = 0
        while idx != len(self.observation_dim):
            if observation[idx] < self.observation_dim[idx] - 1:
                for idx2 in range(idx):
                    observation[idx2] = 0
                observation[idx] += 1
                idx = 0
                observations.append(observation)
                observation = np.copy(observation)
            else:
                idx += 1

        return observations

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def reset(self, deterministic=False, seed=0):
        if deterministic:
            np.random.seed(seed)
        self.current_timestep = 0

    def step(self, action):
        self.current_timestep += 1
        done = (self.current_timestep == self.max_timesteps)
        info = {}
        return None, None, done, info
