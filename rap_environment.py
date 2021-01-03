from functools import reduce
import numpy as np
import gym
from gym import spaces

from MDP import MDPBuilder


class ResourceAllocationEnvironmentBase(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, ra_problem, max_timesteps=500):
        super(ResourceAllocationEnvironmentBase, self).__init__()
        self.ra_problem = ra_problem
        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        self.action_space = spaces.MultiBinary(ra_problem.get_task_count())
        resource_observation_dim = ra_problem.get_max_resource_availabilities() + 1
        new_task_observation_dim = np.ones(ra_problem.get_task_count()) + 1
        running_task_observation_dim = np.ones(ra_problem.get_task_count()) + 1
        self.observation_dim = np.append(
            np.append(new_task_observation_dim, running_task_observation_dim), resource_observation_dim
        )
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

        #print("Last action: ", self.last_action)
        #print("Last Reward: ", self.last_reward)
        #print("Last departed: ", self.last_departed)
        #print("Resources available: ", self.resources_available)
        #print("Tasks in processing: ", self.tasks_in_processing)
        #print("Arrivals: ", self.arrivals)

    def reset(self):
        self.current_timestep = 0

    def step(self, action):
        self.current_timestep += 1
        done = (self.current_timestep == self.max_timesteps)
        info = {}
        return None, None, done, info


class ResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, max_timesteps=500):
        super(ResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.current_resource_availabilities = self.ra_problem.get_max_resource_availabilities()
        self.tasks_in_processing = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.tasks_waiting = np.zeros(self.ra_problem.get_task_count()).astype(int)

    # GETTERS ----------------------------------------------------------------------------------------------------------
    def get_current_resource_availabilities(self):
        return self.current_resource_availabilities

    # SETTERS ----------------------------------------------------------------------------------------------------------
    def set_current_resource_availabilities(self, resource_availabilities):
        assert (resource_availabilities >= 0).all()
        assert (resource_availabilities <= self.ra_problem.get_max_resource_availabilities()).all()
        self.current_resource_availabilities = resource_availabilities
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_reward(self, allocations):
        return float(np.sum(allocations * self.ra_problem.get_rewards()))

    def create_observation(self):
        new_tasks = self.tasks_waiting
        resource_availabilities = self.current_resource_availabilities
        running_tasks = self.tasks_in_processing
        observation = np.append(np.append(new_tasks, running_tasks), resource_availabilities)
        return observation

    def finished_tasks(self):
        return np.random.binomial(self.tasks_in_processing, self.ra_problem.get_task_departure_p())

    def new_tasks(self):
        return np.random.binomial(1, self.ra_problem.get_task_arrival_p())

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(ResourceAllocationEnvironment, self).reset()
        self.current_resource_availabilities = self.ra_problem.get_max_resource_availabilities()
        self.tasks_in_processing = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.tasks_waiting = self.new_tasks()

        observation = self.create_observation()
        return observation

    def step(self, action):
        reward = 0

        tasks_waiting = self.tasks_waiting

        preliminary_allocation = action.astype(int)

        if (tasks_waiting - preliminary_allocation < 0).any():
            allocations = np.zeros(len(preliminary_allocation)).astype(int)
            reward -= 10
        else:
            allocations = preliminary_allocation & tasks_waiting

        resource_availabilities = self.current_resource_availabilities
        resources_used_by_allocations = self.ra_problem.calculate_resources_used(allocations)

        resources_left = resource_availabilities - resources_used_by_allocations

        if (resources_left < 0).any():
            allocations = np.zeros(len(allocations)).astype(int)
            reward -= 10

        reward += self.calculate_reward(allocations)

        observation = self.timestep(allocations)

        _, _, done, info = super(ResourceAllocationEnvironment, self).step(action)

        return observation, reward, done, info

    def timestep(self, allocations):
        finished_tasks = self.finished_tasks()
        self.tasks_in_processing -= finished_tasks

        resources_used_by_finished_tasks = self.ra_problem.calculate_resources_used(finished_tasks)

        self.set_current_resource_availabilities(
            self.current_resource_availabilities + resources_used_by_finished_tasks
        )
        self.tasks_in_processing += allocations
        resources_used_by_allocated_tasks = self.ra_problem.calculate_resources_used(allocations)
        self.set_current_resource_availabilities(
            self.current_resource_availabilities - resources_used_by_allocated_tasks
        )

        self.tasks_waiting = self.new_tasks()

        return self.create_observation()


class SubResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, max_timesteps=500):
        super(SubResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.rap_mdp = MDPBuilder(ra_problem).build_mdp()

    def reset(self):
        super(SubResourceAllocationEnvironment, self).reset()
        initial_state = self.rap_mdp.reset()
        initial_state = self.flatten_observation(initial_state)
        return initial_state

    def step(self, action):
        observation, reward = self.rap_mdp.step(action)
        _, _, done, info = super(SubResourceAllocationEnvironment, self).step(action)
        observation = self.flatten_observation(observation)
        return observation, reward, done, info

    def flatten_observation(self, observation):
        flat_obs = reduce(lambda x, y: x + y, observation)
        return np.array(flat_obs)
