from functools import reduce
import itertools
import numpy as np


class Region:

    def __init__(self, task_conditions):
        self.task_conditions = task_conditions
        self.id_to_conditions = {}
        for task_condition in self.task_conditions:
            task_id = task_condition.task_id
            conditions = self.id_to_conditions.get(task_id, [])
            conditions.append(task_condition)
            self.id_to_conditions[task_id] = conditions

    @staticmethod
    def create_regions(rap):
        fixed_ids = []
        fixed_values = []
        fluent_ids = rap["locked_tasks"]
        resource_requirements = rap["resource_requirements"]
        budget = rap["max_resource_availabilities"]
        possible_task_values = Region.find_task_values_within_budget(fixed_ids, fixed_values, fluent_ids,
                                                                     resource_requirements, budget)

        regions = []
        for values in possible_task_values:
            task_conditions = []
            for task_id, value in zip(rap["locked_tasks"], values):
                task_condition = TaskCondition(task_id=task_id, min_value=int(value), max_value=int(value))
                task_conditions.append(task_condition)
            region = Region(task_conditions)
            regions.append(region)
        return regions

    @staticmethod
    def find_task_values_within_budget(fixed_ids, fixed_values, fluent_ids, resource_requirements, budget):
        assert len(fixed_ids) == len(fixed_values)
        assert len(fluent_ids) > 0

        valid_values = []
        if len(fixed_ids) > 0:
            fixed_cost = np.sum(resource_requirements[fixed_ids] * np.array([fixed_values]).T, axis=0)
        else:
            fixed_cost = np.array([0, 0, 0])
        fluent_id = fluent_ids[0]
        value = 0
        within_budget = True
        while within_budget:
            final_costs = fixed_cost + resource_requirements[fluent_id]*value
            if (final_costs > budget).any():
                within_budget = False
            elif len(fluent_ids) == 1:
                values = np.append(fixed_values, value)
                valid_values.append(values)
            else:
                new_fixed_ids = fixed_ids + [fluent_ids[0]]
                new_fixed_values = fixed_values + [value]
                new_fluent_ids = fluent_ids[1:]
                values = Region.find_task_values_within_budget(new_fixed_ids, new_fixed_values, new_fluent_ids,
                                                               resource_requirements, budget)
                valid_values += values
            value += 1

        return valid_values

    def find_task_values_within_region(self):
        value_lists = []
        task_ids = list(self.id_to_conditions.keys())
        task_ids.sort()
        for task_id in task_ids:
            conditions = self.id_to_conditions[task_id]
            min_value, max_value = reduce(
                lambda old_t, condition: (max(old_t[0], condition.min_value), min(old_t[1], condition.max_value)),
                conditions,
                (-1, 10**6)
            )
            task_values = list(range(min_value, max_value + 1))
            value_lists.append(task_values)
        return list(itertools.product(*value_lists))

    def inside_region(self, task):
        return all([task_condition.satisfied(task) for task_condition in self.task_conditions])

    def distance_from_region(self, tasks):
        distance_vector = np.asarray([task_condition.distance(tasks) for task_condition in self.task_conditions])
        distance = np.linalg.norm(distance_vector)
        return distance

    def task_meets_all_conditions(self, n, task_id):
        conditions = self.id_to_conditions[task_id]
        for condition in conditions:
            if not condition.check(n):
                return False
        return True


class TaskCondition:

    def __init__(self, task_id=0, min_value=0, max_value=1):
        self.task_id = task_id
        self.min_value = min_value
        self.max_value = max_value

    def satisfied(self, tasks):
        return self.min_value <= tasks[self.task_id] <= self.max_value

    def distance(self, tasks):
        if self.min_value <= tasks[self.task_id] <= self.max_value:
            d = 0
        elif tasks[self.task_id] < self.min_value:
            d = self.min_value - tasks[self.task_id]
        else:
            d = tasks[self.task_id] - self.max_value
        return d

    def check(self, n):
        return self.min_value <= n <= self.max_value
