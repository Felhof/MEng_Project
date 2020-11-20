import numpy as np
import gym
from gym import spaces


class ResourceManager(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, ra_problem, max_timesteps=500):
        """
        """
        super(ResourceManager, self).__init__()
        self.ra_problem = ra_problem
        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        self.action_space = spaces.MultiBinary(ra_problem.get_task_count())
        resource_observation_dim = ra_problem.get_max_resource_availabilities() + 1
        new_task_observation_dim = np.ones(ra_problem.get_task_count()) + 1
        self.observation_space = spaces.MultiDiscrete(
            np.append(resource_observation_dim, new_task_observation_dim)
        )

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.current_timestep = 0
        self.ra_problem.reset()
        return self.create_observation()

    def step(self, action):

        tasks_waiting = self.ra_problem.get_tasks_waiting()
        allocations = action.astype(int) & tasks_waiting

        resource_availabilities = self.ra_problem.get_current_resource_availabilities()
        resources_used_by_allocations = self.ra_problem.calculate_resources_used(allocations)

        resources_left = resource_availabilities - resources_used_by_allocations

        if (resources_left < 0).any():
            allocations = np.zeros(len(allocations)).astype(int)

        reward = float(np.sum(allocations * self.ra_problem.get_rewards()))

        self.ra_problem.timestep(allocations)

        observation = self.create_observation()

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

    def create_observation(self):
        new_tasks = self.ra_problem.get_tasks_waiting()
        resource_availabilities = self.ra_problem.get_current_resource_availabilities()
        observation = np.append(resource_availabilities, new_tasks)
        return observation
