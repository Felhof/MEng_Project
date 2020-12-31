import numpy as np
import gym
from gym import spaces


class ResourceAllocationEnvironment(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, ra_problem, max_timesteps=500):
        """
        """
        super(ResourceAllocationEnvironment, self).__init__()
        self.ra_problem = ra_problem
        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        self.action_space = spaces.MultiBinary(ra_problem.get_task_count())
        resource_observation_dim = ra_problem.get_max_resource_availabilities() + 1
        new_task_observation_dim = np.ones(ra_problem.get_task_count()) + 1
        running_task_observation_dim = np.ones(ra_problem.get_task_count()) + 1
        self.observation_dim = np.append(
            resource_observation_dim, np.append(new_task_observation_dim, running_task_observation_dim)
        )
        self.observation_space = spaces.MultiDiscrete(
            self.observation_dim
        )

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.current_timestep = 0
        observation = self.ra_problem.reset()
        return observation

    def step(self, action):
        reward = 0

        tasks_waiting = self.ra_problem.get_tasks_waiting()

        preliminary_allocation = action.astype(int)

        if (tasks_waiting - preliminary_allocation < 0).any():
            allocations = np.zeros(len(preliminary_allocation)).astype(int)
            reward -= 10
        else:
            allocations = preliminary_allocation & tasks_waiting

        resource_availabilities = self.ra_problem.get_current_resource_availabilities()
        resources_used_by_allocations = self.ra_problem.calculate_resources_used(allocations)

        resources_left = resource_availabilities - resources_used_by_allocations

        if (resources_left < 0).any():
            allocations = np.zeros(len(allocations)).astype(int)
            reward -= 10

        reward += self.ra_problem.calculate_reward(allocations)

        observation = self.ra_problem.timestep(allocations)

        self.current_timestep += 1
        done = (self.current_timestep == self.max_timesteps)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return observation, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        #print("Last action: ", self.last_action)
        #print("Last Reward: ", self.last_reward)
        #print("Last departed: ", self.last_departed)
        #print("Resources available: ", self.resources_available)
        #print("Tasks in processing: ", self.tasks_in_processing)
        #print("Arrivals: ", self.arrivals)

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
