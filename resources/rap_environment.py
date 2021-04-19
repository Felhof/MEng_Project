from functools import reduce
import numpy as np
import gym
from gym import spaces
import torch

from resources.MDP import MDPBuilder


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

        self.build_action_and_observation_space(resource_observation_dim, new_task_observation_dim,
                                                running_task_observation_dim)

    def build_action_and_observation_space(self, resource_observation_dim, new_task_observation_dim,
                                           running_task_observation_dim):
        self.action_space = spaces.MultiBinary(self.ra_problem.get_task_count())
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
        self.current_state = None

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

    def update_current_state(self):
        new_tasks = self.tasks_waiting
        resource_availabilities = self.current_resource_availabilities
        running_tasks = self.tasks_in_processing
        self.current_state = np.append(np.append(new_tasks, running_tasks), resource_availabilities)

    def finished_tasks(self):
        departure_probabilities = self.ra_problem.get_task_departure_p()
        return np.random.binomial(self.tasks_in_processing, departure_probabilities)

    def new_tasks(self):
        return np.random.binomial(1, self.ra_problem.get_task_arrival_p())

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(ResourceAllocationEnvironment, self).reset()
        self.current_resource_availabilities = self.max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.tasks_waiting = self.new_tasks()

        self.update_current_state()
        return self.current_state

    def step(self, action):
        tasks_waiting = self.tasks_waiting

        preliminary_allocation = action.astype(int)

        if (tasks_waiting - preliminary_allocation < 0).any():
            reward = 0
        else:
            allocations = preliminary_allocation & tasks_waiting

            resource_availabilities = self.current_resource_availabilities
            resources_used_by_allocations = self.ra_problem.calculate_resources_used(allocations)

            resources_left = resource_availabilities - resources_used_by_allocations

            if (resources_left < 0).any():
                reward = 0
            else:
                self.timestep(allocations)
                self.update_current_state()
                reward = self.calculate_reward(allocations)

        observation = self.current_state

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


class RestrictedResourceAllocationEnvironment(ResourceAllocationEnvironment):

    def __init__(self, ra_problem, task_locks=None, lower_lvl_models=None, max_timesteps=500):
        self.restricted_tasks = list(task_locks.keys())
        self.amount_of_locked_tasks = list(task_locks.values())
        super(RestrictedResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.lower_lvl_models = lower_lvl_models
        self.in_hull = False

        locked_tasks = np.zeros(self.ra_problem.get_task_count())
        for task, locked_amount in task_locks.items():
            locked_tasks[task] = min(locked_amount)

        cost_of_restricted_tasks = self.ra_problem.calculate_resources_used(locked_tasks)
        self.max_resource_availabilities = self.ra_problem.get_max_resource_availabilities() - cost_of_restricted_tasks

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(RestrictedResourceAllocationEnvironment, self).reset()
        self.tasks_in_processing[self.restricted_tasks] = [np.random.choice(r) for r in self.amount_of_locked_tasks]
        self.update_current_state()

        self.in_hull = False

        return self.current_state

    def finished_tasks(self):
        departure_probabilities = self.ra_problem.get_task_departure_p()
        finished_tasks = np.random.binomial(self.tasks_in_processing, departure_probabilities)

        def out_of_range(task_idx):
            running = self.tasks_in_processing[task_idx] - finished_tasks[task_idx]
            idx = self.restricted_tasks.index(task_idx)
            return not(self.amount_of_locked_tasks[idx][0] <= running <= self.amount_of_locked_tasks[idx][-1])

        reset_idxs = list(filter(out_of_range, self.restricted_tasks))
        finished_tasks[reset_idxs] = 0

        return finished_tasks

    def calculate_reward(self, allocations):
        model_key = tuple(self.tasks_in_processing[self.restricted_tasks])
        lower_lvl_model = self.lower_lvl_models.get(model_key, None)
        if lower_lvl_model is not None:
            self.in_hull = True
            state_tensor = torch.tensor(self.current_state).unsqueeze(0)
            _, value, _ = lower_lvl_model.policy.forward(state_tensor)
            reward = value.item() * (1 - self.current_timestep/self.max_timesteps)
        else:
            reward = float(np.sum(allocations * self.ra_problem.get_rewards()))
        return reward

    def step(self, action):
        observation, reward, done, info = super(RestrictedResourceAllocationEnvironment, self).step(action)
        done = done or self.in_hull

        return observation, reward, done, info


class MDPResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, max_timesteps=500):
        super(MDPResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.rap_mdp = MDPBuilder(ra_problem).build_mdp()

    def reset(self):
        super(MDPResourceAllocationEnvironment, self).reset()
        initial_state = self.rap_mdp.reset()
        initial_state = self.flatten_observation(initial_state)
        return initial_state

    def step(self, action):
        observation, reward = self.rap_mdp.step(action)
        _, _, done, info = super(MDPResourceAllocationEnvironment, self).step(action)
        observation = self.flatten_observation(observation)
        return observation, reward, done, info

    def flatten_observation(self, observation):
        flat_obs = reduce(lambda x, y: x + y, observation)
        return np.array(flat_obs)


class RestrictedMDPResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, restricetd_mdp, max_timesteps=500):
        super(RestrictedMDPResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.rmdp = restricetd_mdp

    def reset(self):
        super(RestrictedMDPResourceAllocationEnvironment, self).reset()
        initial_state = self.rmdp.reset()
        initial_state = self.flatten_observation(initial_state)
        return initial_state

    def step(self, action):
        observation, reward, reached_hull = self.rmdp.step(action)
        _, _, no_time_left, info = super(RestrictedMDPResourceAllocationEnvironment, self).step(action)
        observation = self.flatten_observation(observation)
        done = reached_hull or no_time_left
        return observation, reward, done, info

    def flatten_observation(self, observation):
        flat_obs = reduce(lambda x, y: x + y, observation)
        return np.array(flat_obs)
