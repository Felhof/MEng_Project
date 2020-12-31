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
        self.current_resource_availabilities = max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count).astype(int)
        self.tasks_waiting = np.zeros(self.task_count).astype(int)

        # Note to self: When not using urr assumption must take resource requirement into account
        self.expected_rewards = [
            (task_number, reward * departure_p)
            for
            task_number, (reward, departure_p)
            in
            enumerate(zip(rewards, self.tasks_departure_p))
        ]

        self.expected_rewards.sort(key=lambda x: -x[1])

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

    def get_resource_count(self):
        return self.resource_count

    def get_rewards(self):
        return self.rewards

    def get_resource_requirements(self):
        return self.resource_requirements

    def reset(self):
        self.current_resource_availabilities = self.max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count).astype(int)
        self.tasks_waiting = self.new_tasks()

    def timestep(self, allocations):
        """
        :param allocations: ([int]) length M list, for each tasks if resources are allocated to it or not
        """
        finished_tasks = self.finished_tasks()
        self.tasks_in_processing -= finished_tasks

        resources_used_by_finished_tasks = self.calculate_resources_used(finished_tasks)

        self.set_current_resource_availabilities(
            self.current_resource_availabilities + resources_used_by_finished_tasks
        )
        self.tasks_in_processing += allocations
        resources_used_by_allocated_tasks = self.calculate_resources_used(allocations)
        self.set_current_resource_availabilities(
            self.current_resource_availabilities - resources_used_by_allocated_tasks
        )

        self.tasks_waiting = self.new_tasks()

    def new_tasks(self):
        return np.random.binomial(1, self.tasks_arrival_p)

    def finished_tasks(self):
        return np.random.binomial(self.tasks_in_processing, self.tasks_departure_p)

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

    def create_observation(self):
        new_tasks = self.get_tasks_waiting()
        resource_availabilities = self.get_current_resource_availabilities()
        running_tasks = self.tasks_in_processing
        observation = np.append(resource_availabilities, np.append(new_tasks, running_tasks))
        return observation

    def calculate_reward(self, allocations):
        return float(np.sum(allocations * self.get_rewards()))
