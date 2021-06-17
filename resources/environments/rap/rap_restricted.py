import numpy as np
import torch
from gym import spaces

from resources.environments.rap.rap_environment import ResourceAllocationEnvironment


class RegionalResourceAllocationEnvironment(ResourceAllocationEnvironment):

    def __init__(self, ra_problem, region=None, max_timesteps=500):
        self.region = region
        self.restricted_task_ids = [task_lock.task_id for task_lock in region.task_conditions]
        self.locked_task_ranges = [list(range(task_lock.min_value, task_lock.max_value + 1))
                                   for task_lock in region.task_conditions]
        super(RegionalResourceAllocationEnvironment, self).__init__(ra_problem, idle_reward=0,
                                                                       max_timesteps=max_timesteps)

        locked_tasks = np.zeros(self.ra_problem.get_task_count())
        for task_lock in region.task_conditions:
            locked_tasks[task_lock.task_id] = task_lock.min_value

        self.min_cost_of_restricted_tasks = self.ra_problem.calculate_resources_used(locked_tasks)
        self.max_resource_availabilities = self.ra_problem.get_max_resource_availabilities() - \
                                           self.min_cost_of_restricted_tasks

    def reset(self, deterministic=False, seed=0):
        super(RegionalResourceAllocationEnvironment, self).reset(deterministic=deterministic, seed=seed)
        self.tasks_in_processing[self.restricted_task_ids] = [min(r) for r in self.locked_task_ranges]


class AbbadDaouiRegionalResourceAllocationEnvironment(RegionalResourceAllocationEnvironment):

    def __init__(self, ra_problem, region=None, lower_lvl_models=None, max_timesteps=500):
        super(AbbadDaouiRegionalResourceAllocationEnvironment, self).__init__(ra_problem, region=region,
                                                                              max_timesteps=max_timesteps)
        self.lower_lvl_models = lower_lvl_models
        self.in_hull = False
        print(3)

    def reset(self, deterministic=False, seed=0):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(AbbadDaouiRegionalResourceAllocationEnvironment, self).reset(deterministic=deterministic, seed=seed)
        cost_of_tasks = self.ra_problem.calculate_resources_used(self.tasks_in_processing)
        cost_of_tasks -= self.min_cost_of_restricted_tasks.astype(int)
        self.set_current_resource_availabilities(
            self.get_current_resource_availabilities() - cost_of_tasks
        )
        self.update_current_state()
        self.in_hull = False

        return self.current_state

    def finished_tasks(self):
        departure_probabilities = self.ra_problem.get_task_departure_p()
        finished_tasks = np.random.binomial(self.tasks_in_processing, departure_probabilities)

        def out_of_range(task_idx):
            n = self.tasks_in_processing[task_idx]
            return not self.region.task_meets_all_conditions(n, task_idx)

        reset_idxs = list(filter(out_of_range, self.restricted_task_ids))
        finished_tasks[reset_idxs] = 0

        return finished_tasks

    def calculate_reward(self, allocations):
        model_key = tuple(self.tasks_in_processing[self.restricted_task_ids])
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
        observation, reward, done, info = super(AbbadDaouiRegionalResourceAllocationEnvironment, self).step(action)
        done = done or self.in_hull

        return observation, reward, done, info


class DeanLinRegionalResourceAllocationEnvironment(RegionalResourceAllocationEnvironment):

    def __init__(self, ra_problem, region=None, max_timesteps=500):
        super(DeanLinRegionalResourceAllocationEnvironment, self).__init__(ra_problem, region=region,
                                                                           max_timesteps=max_timesteps)

        self.number_of_locked_tasks = len(self.restricted_task_ids)
        self.action_space = spaces.MultiBinary(self.ra_problem.get_task_count() - self.number_of_locked_tasks)
        self.observation_dim = self.translate_state(self.observation_dim)
        self.observation_space = spaces.MultiDiscrete(self.observation_dim)

    def translate_state(self, state):
        length = len(state) // 2
        arrivals = state[:length]
        running = state[length:]
        state = np.append(arrivals[:-self.number_of_locked_tasks], running[:-self.number_of_locked_tasks])
        return state

    def update_current_state(self):
        super(DeanLinRegionalResourceAllocationEnvironment, self).update_current_state()
        self.current_state = self.translate_state(self.current_state)

    def finished_tasks(self):
        finished_tasks = super(DeanLinRegionalResourceAllocationEnvironment, self).finished_tasks()
        finished_tasks[self.restricted_task_ids] = 0

        return finished_tasks

    def reset(self, deterministic=False, seed=0):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(DeanLinRegionalResourceAllocationEnvironment, self).reset(deterministic=deterministic, seed=seed)
        cost_of_tasks = self.ra_problem.calculate_resources_used(self.tasks_in_processing)
        cost_of_tasks -= self.min_cost_of_restricted_tasks.astype(int)
        self.set_current_resource_availabilities(
            self.current_resource_availabilities - cost_of_tasks
        )
        self.update_current_state()

        return self.current_state

    def step(self, action):
        action = np.append(action, np.array([0] * self.number_of_locked_tasks))
        observation, reward, done, info = super(DeanLinRegionalResourceAllocationEnvironment, self).step(action)

        return observation, reward, done, info