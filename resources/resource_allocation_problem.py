import numpy as np


class ResourceAllocationProblem:
    def __init__(self, rewards, resource_requirements, max_resource_availabilities, tasks_arrival_p, tasks_departure_p):
        """
        :param rewards: ([int]) length M list, reward for each task
        :param resource_requirements: ([[float]]) MxK, where [a][b] gives the amount of resource b required for task a
        :param max_resource_availabilities: ([float]) length K list, how much of each resource is available
        :param tasks_arrival_p: ([float]) length M list, probability of task arriving each time step
        :param tasks_departure_p: ([float]) length M list, probability of task finishing processing each time step
        M : number of tasks
        K : number of resources
        """

        self.rewards = rewards
        self.resource_requirements = resource_requirements
        self.max_resource_availabilities = max_resource_availabilities
        self.tasks_arrival_p = tasks_arrival_p
        self.tasks_departure_p = tasks_departure_p

        self.task_count = len(rewards)
        self.resource_count = len(max_resource_availabilities)

        # Note to self: When not using urr assumption must take resource requirement into account
        self.expected_rewards = [
            (task_number, reward * departure_p)
            for
            task_number, (reward, departure_p)
            in
            enumerate(zip(rewards, self.tasks_departure_p))
        ]

        self.expected_rewards.sort(key=lambda x: -x[1])

    # GETTERS ----------------------------------------------------------------------------------------------------------
    def get_max_resource_availabilities(self):
        return self.max_resource_availabilities

    def get_resource_count(self):
        return self.resource_count

    def get_resource_requirements(self):
        return self.resource_requirements

    def get_rewards(self):
        return self.rewards

    def get_task_arrival_p(self):
        return self.tasks_arrival_p

    def get_task_count(self):
        return self.task_count

    def get_task_departure_p(self):
        return self.tasks_departure_p
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_resources_used(self, tasks):
        resources_used = tasks * np.transpose(self.resource_requirements)
        return resources_used.sum(axis=1)

    def get_heuristic_solution(self, observation):
        free_resources = observation[0, :self.resource_count]
        tasks_arrivals = observation[0, self.resource_count:]

        allocations = np.zeros(self.task_count)

        for task_number, expected_reward in self.expected_rewards:
            if not tasks_arrivals[task_number]:
                continue
            resources_remaining = free_resources - self.resource_requirements[task_number]
            if (resources_remaining >= 0).all():
                free_resources = resources_remaining
                allocations[task_number] = 1

        return np.array([allocations])


small_problem = {
    "rewards": np.array([3, 2, 1]),
    "resource_requirements": np.array([[1, 0], [0, 1], [1, 1]]),
    "max_resource_availabilities": np.array([2, 3]),
    "task_arrival_p": np.array([0.3, 0.4, 0.9]),
    "task_departure_p": np.array([0.6, 0.6, 0.9]),
}


