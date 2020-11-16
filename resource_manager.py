import numpy as np
import gym
from gym import spaces
from scipy.stats import bernoulli, binom


class ResourceManager(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, resource_count=3, task_count=3, rewards=[3, 2, 1], resource_limit=5,
                 task_arrival_p=[0.5, 0.5, 0.5], task_departure_p=[0.3, 0.3, 0.3], max_timesteps=500):
        """
        :param K: (int) amount of resources
        :param M: (int) amount of tasks
        :param U: (int) rewards for tasks
        :param resource_availability: (int) how often a resource can be allocated
        """
        super(ResourceManager, self).__init__()

        self.last_action = []
        self.last_departed = [0, 0, 0]
        self.last_reward = 0

        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        self.resource_limit = resource_limit
        self.rewards = rewards
        self.task_count = task_count
        self.resource_count = resource_count
        self.task_arrival_p = task_arrival_p
        self.task_departure_p = task_departure_p
        self.resources_available = resource_limit
        self.tasks_in_processing = np.array([0] * task_count)
        self.arrivals = [bernoulli.rvs(p) for p in self.task_arrival_p]

        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(task_count,), dtype=np.int)

        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(task_count + 1,), dtype=np.int)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.current_timestep = 0

        self.resources_available = self.resource_limit
        self.tasks_in_processing = np.array([0] * self.task_count)

        self.arrivals = [bernoulli.rvs(p) for p in self.task_arrival_p]
        return np.array(self.arrivals + [self.resources_available])

    def step(self, action):

        reward = 0
        new_tasks = np.array([0]*self.task_count)
        resource_cost = 0

        for idx, (allocation, task) in enumerate(zip(list(action), self.arrivals)):
            if allocation == 1 and task == 1 and self.resources_available > 0:
                new_tasks[idx] += 1
                resource_cost += 1
                reward += self.rewards[idx]

        for idx, count in enumerate(self.tasks_in_processing):
            departing = binom.rvs(count, self.task_departure_p[idx])
            self.last_departed[idx] = departing
            self.tasks_in_processing[idx] -= departing
            self.resources_available += departing

        self.tasks_in_processing += new_tasks
        self.resources_available -= resource_cost
        self.arrivals = [bernoulli.rvs(p) for p in self.task_arrival_p]

        observation = np.array(self.arrivals + [self.resources_available])

        self.last_action = action
        self.last_reward = reward

        self.current_timestep += 1
        done = (self.current_timestep == self.max_timesteps)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return observation, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("Last action: ", self.last_action)
        print("Last Reward: ", self.last_reward)
        print("Last departed: ", self.last_departed)
        print("Resources available: ", self.resources_available)
        print("Tasks in processing: ", self.tasks_in_processing)
        print("Arrivals: ", self.arrivals)

    def close(self):
        pass
