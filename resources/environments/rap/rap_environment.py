import numpy as np

from resources.environments.rap.rap_base import ResourceAllocationEnvironmentBase


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
        running_tasks = self.tasks_in_processing
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


