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
        #self.observation_dim = np.append(
        #    np.append(new_task_observation_dim, running_task_observation_dim), resource_observation_dim
        #)
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


class ResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, idle_reward=0, max_timesteps=500):
        super(ResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.current_resource_availabilities = self.ra_problem.get_max_resource_availabilities()
        self.tasks_in_processing = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.tasks_waiting = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.current_state = None
        self.idle_reward = idle_reward

    # GETTERS ----------------------------------------------------------------------------------------------------------
    def get_current_resource_availabilities(self):
        return self.current_resource_availabilities

    # SETTERS ----------------------------------------------------------------------------------------------------------
    def set_current_resource_availabilities(self, resource_availabilities):
        assert (resource_availabilities >= 0).all(), "must have nonnegative resources"
        if not (resource_availabilities <= self.ra_problem.get_max_resource_availabilities()).all():
            print(resource_availabilities)
        assert (resource_availabilities <= self.ra_problem.get_max_resource_availabilities()).all(), "resources must be within limit"
        self.current_resource_availabilities = resource_availabilities
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_reward(self, allocations):
        return float(np.sum(allocations * self.ra_problem.get_rewards()))

    def update_current_state(self):
        new_tasks = self.tasks_waiting
        resource_availabilities = self.current_resource_availabilities
        running_tasks = self.tasks_in_processing
        #self.current_state = np.append(np.append(new_tasks, running_tasks), resource_availabilities)
        self.current_state = np.append(new_tasks, running_tasks)

    def finished_tasks(self):
        departure_probabilities = self.ra_problem.get_task_departure_p()
        return np.random.binomial(self.tasks_in_processing, departure_probabilities)

    def new_tasks(self):
        return np.random.binomial(1, self.ra_problem.get_task_arrival_p())

    def reset(self, deterministic=False, seed=0):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super(ResourceAllocationEnvironment, self).reset(deterministic=deterministic, seed=seed)
        self.current_resource_availabilities = self.max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.ra_problem.get_task_count()).astype(int)
        self.tasks_waiting = self.new_tasks()

        self.update_current_state()
        return self.current_state

    def step(self, action):
        tasks_waiting = self.tasks_waiting

        preliminary_allocation = action.astype(int)

        if (tasks_waiting - preliminary_allocation < 0).any():
            preliminary_allocation = np.zeros(len(preliminary_allocation), dtype=int)

        allocations = preliminary_allocation & tasks_waiting

        resource_availabilities = self.current_resource_availabilities
        resources_used_by_allocations = self.ra_problem.calculate_resources_used(allocations)

        resources_left = resource_availabilities - resources_used_by_allocations

        if (resources_left < 0).any():
            allocations = np.zeros(len(allocations), dtype=int)

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
        self.tasks_in_processing[self.restricted_task_ids] = [np.random.choice(r) for r in self.locked_task_ranges]


class AbbadDaouiRegionalResourceAllocationEnvironment(RegionalResourceAllocationEnvironment):

    def __init__(self, ra_problem, region=None, lower_lvl_models=None, max_timesteps=500):
        #print(2)
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
        #print("reset")
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
        #print("finished task")
        departure_probabilities = self.ra_problem.get_task_departure_p()
        finished_tasks = np.random.binomial(self.tasks_in_processing, departure_probabilities)

        def out_of_range(task_idx):
            n = self.tasks_in_processing[task_idx]
            return not self.region.task_meets_all_conditions(n, task_idx)

        reset_idxs = list(filter(out_of_range, self.restricted_task_ids))
        finished_tasks[reset_idxs] = 0

        return finished_tasks

    def calculate_reward(self, allocations):
        #print("rewards")
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
        #print("steps")
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


class MDPResourceAllocationEnvironment(ResourceAllocationEnvironmentBase):
    def __init__(self, ra_problem, max_timesteps=500):
        super(MDPResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.rap_mdp = MDPBuilder(ra_problem).build_mdp()

    def reset(self, deterministic=False, seed=0):
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

    def reset(self, deterministic=False, seed=0):
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
