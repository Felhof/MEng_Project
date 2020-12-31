import numpy as np

from MDP import MarkovDecisionProcess


class ResourceAllocationProblemBase:
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

    # METHODS TO BE IMPLEMENTED BY CHILD CLASSES -----------------------------------------------------------------------
    def reset(self):
        return NotImplementedError

    def timestep(self, allocations):
        """
        :param allocations: ([int]) length M list, for each tasks if resources are allocated to it or not
        """
        return NotImplementedError


class ResourceAllocationProblem(ResourceAllocationProblemBase):

    def __init__(self, rewards, resource_requirements, max_resource_availabilities, tasks_arrival_p, tasks_departure_p):
        super(ResourceAllocationProblem, self).__init__(rewards, resource_requirements, max_resource_availabilities,
                                                        tasks_arrival_p, tasks_departure_p)
        self.current_resource_availabilities = max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count).astype(int)
        self.tasks_waiting = np.zeros(self.task_count).astype(int)

    # GETTERS ----------------------------------------------------------------------------------------------------------
    def get_current_resource_availabilities(self):
        return self.current_resource_availabilities

    # SETTERS ----------------------------------------------------------------------------------------------------------
    def set_current_resource_availabilities(self, resource_availabilities):
        assert (resource_availabilities >= 0).all()
        assert (resource_availabilities <= self.max_resource_availabilities).all()
        self.current_resource_availabilities = resource_availabilities
    # ------------------------------------------------------------------------------------------------------------------

    # IMPLEMENTATIONS OF PARENT CLASS METHODS --------------------------------------------------------------------------
    def reset(self):
        self.current_resource_availabilities = self.max_resource_availabilities
        self.tasks_in_processing = np.zeros(self.task_count).astype(int)
        self.tasks_waiting = self.new_tasks()

        return self.create_observation()

    def timestep(self, allocations):
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

        return self.create_observation()
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_reward(self, allocations):
        return float(np.sum(allocations * self.get_rewards()))

    def create_observation(self):
        new_tasks = self.tasks_waiting
        resource_availabilities = self.get_current_resource_availabilities()
        running_tasks = self.tasks_in_processing
        observation = np.append(resource_availabilities, np.append(new_tasks, running_tasks))
        return observation

    def finished_tasks(self):
        return np.random.binomial(self.tasks_in_processing, self.tasks_departure_p)

    def new_tasks(self):
        return np.random.binomial(1, self.tasks_arrival_p)


def bit_permutations(n):
    if n == 1:
        return [[0], [1]]
    else:
        sequences = []
        shorter_sequences = bit_permutations(n - 1)
        for s in shorter_sequences:
            sequences.append(s + [0])
            sequences.append(s + [1])
        return sequences


class MDPBuilder:

    def __init__(self, resource_allocation_problem):
        self.rap = resource_allocation_problem
        self.possible_departing_tasks = {}

    def all_states(self, arrived_task_states):
        states = []

        for running_tasks, available_resources in self.running_tasks_and_available_resource_states():
            for arrived_tasks in arrived_task_states:
                arrived_tasks = tuple(arrived_tasks)
                state = (arrived_tasks, running_tasks, available_resources)
                states.append(state)

        return states

    def arrived_task_states(self):
        states = bit_permutations(self.rap.get_task_count())
        return states

    def build_mdp(self):
        actions = self.arrived_task_states()
        states = self.all_states(actions)
        transitions = {}
        arrival_probabilities = {tuple(action): self.get_arrival_probabilities(action) for action in actions}

        for state in states:
            waiting_tasks, running_tasks, available_resources = state
            waiting_tasks = np.array(waiting_tasks)
            running_tasks = np.array(running_tasks)
            available_resources = np.array(available_resources)
            for action in actions:
                action = np.array(action)
                tasks_after_allocation = running_tasks + action
                resource_requirement_after_allocation = self.rap.calculate_resources_used(tasks_after_allocation)
                if not((waiting_tasks - action >= 0).all() and (resource_requirement_after_allocation <= available_resources).all()):
                    continue
                state_action_transitions = []
                reward = np.sum(action * self.rap.get_rewards())
                possible_departing_tasks = self.get_possible_departing_tasks(running_tasks)
                for departing_tasks in possible_departing_tasks:
                    departure_probability = np.prod(self.rap.get_tasks_departure_p() ** departing_tasks)
                    remaining_tasks = tasks_after_allocation - departing_tasks
                    new_resource_requirement = self.rap.calculate_resources_used(remaining_tasks)
                    for arriving_tasks in actions:
                        successor_state = (arriving_tasks, departing_tasks, new_resource_requirement)
                        arrival_probability = arrival_probabilities[arriving_tasks]
                        transition_probability = departure_probability * arrival_probability
                        transition = (successor_state, transition_probability, reward)
                        state_action_transitions.append(transition)
                transitions[(state, tuple(action))] = state_action_transitions

        mdp = MarkovDecisionProcess(states, actions, transitions)
        return mdp

    def get_arrival_probabilities(self, arrivals):
        arrival_probs = np.prod(arrivals * self.rap.get_tasks_arrival_p())
        no_arrival_probs = np.prod((1 - arrivals) * (1 - self.rap.get_tasks_arrival_p()))
        return arrival_probs * no_arrival_probs

    def get_possible_departing_tasks(self, running_task):
        tasks_departing = self.possible_departing_tasks.get(tuple(running_task), None)
        if tasks_departing is not None:
            return tasks_departing
        if sum(running_task) == 1 or sum(running_task) == 0:
            self.possible_departing_tasks[tuple(running_task)] = [running_task]
            return [running_task]

        all_tasks = np.copy(running_task)
        departing_task_list = [all_tasks]
        for idx in range(self.rap.get_task_count()):
            tasks = np.copy(running_task)
            if running_task[idx] > 0:
                tasks[idx] -= 1
                departing_task_list += self.get_possible_departing_tasks(tasks)

        self.possible_departing_tasks[tuple(running_task)] = departing_task_list
        return departing_task_list

    def running_tasks_and_available_resource_states(self):
        running_tasks = np.zeros(self.rap.get_task_count(), dtype="int32")
        available_resources = np.array(self.rap.get_max_resource_availabilities())
        initial_state = (tuple(running_tasks), tuple(available_resources))

        discovered = {}
        stack = [initial_state]

        while len(stack) > 0:
            state = stack.pop()

            if not discovered.get(state, False):
                yield state
                discovered[state] = True
                running_tasks = state[0]
                available_resources = state[1]
                for idx in range(self.rap.get_task_count()):
                    new_running_tasks = np.copy(running_tasks)
                    new_available_resources = np.copy(available_resources)
                    new_running_tasks[idx] += 1
                    resources_remaining = new_available_resources - self.rap.get_resource_requirements()[idx]
                    if (resources_remaining >= 0).all():
                        new_state = (tuple(new_running_tasks), tuple(resources_remaining))
                        stack.append(new_state)


class SubResourceAllocationProblem(ResourceAllocationProblemBase):

    def __init__(self, rewards, resource_requirements, max_resource_availabilities, tasks_arrival_p, tasks_departure_p):
        super(SubResourceAllocationProblem, self).__init__(rewards, resource_requirements, max_resource_availabilities,
                                                           tasks_arrival_p, tasks_departure_p)
        self.rap_mdp = MDPBuilder(self).build_mdp()


small_problem = {
    "rewards": np.array([3, 2, 1]),
    "resource_requirements": np.array([[1, 0], [0, 1], [1, 1]]),
    "max_resource_availabilities": np.array([2, 3]),
    "task_arrival_p": np.array([0.3, 0.4, 0.9]),
    "task_departure_p": np.array([0.6, 0.6, 0.9]),
}

srap = SubResourceAllocationProblem(
    small_problem["rewards"],
    small_problem["resource_requirements"],
    small_problem["max_resource_availabilities"],
    small_problem["task_arrival_p"],
    small_problem["task_departure_p"]
)

for s in srap.all_states():
    print(s)


