from gym import spaces
import numpy as np

from resources.environments.rap_environment import ResourceAllocationEnvironment


class ADPResourceAllocationEnvironment(ResourceAllocationEnvironment):

    def __init__(self, ra_problem, regions, regional_policies, abstract_action_to_direction=None, n_abstract_actions=3,
                 n_locked_tasks=1, max_timesteps=500):
        super(ADPResourceAllocationEnvironment, self).__init__(ra_problem, max_timesteps=max_timesteps)
        self.regions = regions
        self.regional_policies = regional_policies  # Dictionary from regions to list of abstract actions
        self.action_to_direction = abstract_action_to_direction
        self.action_space = spaces.Discrete(n_abstract_actions)
        self.observation_space = spaces.MultiDiscrete([len(regions)] + ([2] * n_locked_tasks))
        self.n_locked_tasks = n_locked_tasks

    def reset(self, deterministic=False, seed=0):
        state = super(ADPResourceAllocationEnvironment, self).reset(deterministic=deterministic, seed=seed)
        observation = self.build_observation(state)
        return observation

    def step(self, action):
        direction = self.action_to_direction[action]
        abstract_action = None
        for id, region in enumerate(self.regions):
            if region.inside_region(self.tasks_in_processing):
                regional_policy = self.regional_policies[id]
                abstract_action = regional_policy.get(direction, regional_policy["Stay"])
                break
        assert abstract_action is not None

        if direction == "Stay":
            lower_level_state = self.translate_state_to_lower_level(self.current_state)
            lower_level_action, _ = abstract_action.predict(lower_level_state, deterministic=True)
            lower_level_action = np.append(lower_level_action, [0] * self.n_locked_tasks)
        else:
            lower_level_action = abstract_action
        lower_level_state, reward, done, info = super(ADPResourceAllocationEnvironment, self).step(lower_level_action)

        observation = self.build_observation(lower_level_state)

        info["lower level state"] = lower_level_state
        info["lower level action"] = lower_level_action

        return observation, reward, done, info

    def translate_state_to_lower_level(self, state):
        length = len(state) // 2
        arrivals = state[:length]
        running = state[length:]
        state = np.append(arrivals[:-self.n_locked_tasks], running[:-self.n_locked_tasks])
        return state

    def build_observation(self, state):
        significant_arrivals = self.get_significant_arrivals(state)
        current_region_id = self.get_current_region_id()
        observation = np.append(current_region_id, significant_arrivals)
        return observation

    def get_current_region_id(self):
        current_region_id = -1
        for id, region in enumerate(self.regions):
            if region.inside_region(self.tasks_in_processing):
                current_region_id = id
        assert current_region_id != -1
        return current_region_id

    def get_significant_arrivals(self, state):
        arrived_tasks_index = len(state) // 2
        significant_arrivals = state[arrived_tasks_index - self.n_locked_tasks:arrived_tasks_index]
        return significant_arrivals