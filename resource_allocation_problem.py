import numpy as np


class ResourceAllocationProblem:

    def __init__(self, rewards, resource_requirements, max_resource_availabilities, task_arrival_p, task_departure_p):
        """
        :param rewards: ([int]) length M list, reward for each task
        :param resource_requirements: ([[float]]) MxK, where [a][b] gives the amount of resource b required for task a
        :param max_resource_availabilities: ([float]) length K list, how much of each resource is available
        :param task_arrival_p: ([float]) length M list, probability of task arriving each time step
        :param task_departure_p: ([float]) length M list, probability of task finishing processing each time step
        M : number of tasks
        K : number of resources
        """

        self.rewards = rewards
        self.resource_requirements = resource_requirements
        self.max_resource_availabilities = max_resource_availabilities
        self.task_arrival_p = task_arrival_p
        self.task_departure_p = task_departure_p

        self.task_count = len(rewards)
        self.resource_count = len(max_resource_availabilities)
        self.current_resource_availabilities = max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count)
        self.tasks_waiting = np.zeros(self.task_count)

    def get_max_resource_availabilities(self):
        return self.max_resource_availabilities

    def get_current_resource_availabilities(self):
        return self.current_resource_availabilities

    def set_current_resource_availabilities(self, resource_availabilities):
        assert (resource_availabilities >= 0).all()
        assert (resource_availabilities <= self.max_resource_availabilities).all()
        self.current_resource_availabilities = resource_availabilities

    def get_tasks_waiting(self):
        return self.tasks_waiting

    def get_task_count(self):
        return self.task_count

    def get_rewards(self):
        return self.rewards

    def get_resource_requirements(self):
        return self.resource_requirements

    def reset(self):
        self.current_resource_availabilities = self.max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count)
        self.tasks_waiting = self.new_tasks()

    def timestep(self, allocations):
        """
        :param allocations: ([int]) length M list, for each tasks if resources are allocated to it or not
        """
        finished_tasks = self.finished_tasks()
        self.tasks_in_processing -= finished_tasks
        self.set_current_resource_availabilities(
            self.current_resource_availabilities + (finished_tasks * self.resource_requirements)
        )
        self.tasks_in_processing += allocations
        self.set_current_resource_availabilities(
            self.current_resource_availabilities - (allocations * self.resource_requirements)
        )

        self.tasks_waiting = self.new_tasks()

    def new_tasks(self):
        return np.random.binomial(self.task_count, self.task_arrival_p)

    def finished_tasks(self):
        return np.random.binomial(self.tasks_in_processing, self.task_departure_p)

