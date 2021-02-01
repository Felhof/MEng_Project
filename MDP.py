from functools import reduce
import itertools
import numpy as np


def build_index(items):
    item_to_idx = {}
    idx_to_item = {}

    for idx, item in enumerate(items):
        item_to_idx[item] = idx
        idx_to_item[idx] = item

    return item_to_idx, idx_to_item


class MarkovDecisionProcess:

    def __init__(self, states, actions, transitions, initial_states):
        self.state_to_idx, self.idx_to_state = build_index(states)
        self.action_to_idx, self.idx_to_action = build_index(actions)
        self.transition_matrix, self.reward_matrix = self.build_transitions(states, actions, transitions)
        self.initial_states_idxs = [self.state_to_idx[initial_state] for initial_state in initial_states]
        self.current_state_idx = self.state_to_idx[self.reset()]

    def build_transitions(self, states, actions, transitions_dict):
        transition_matrix = np.zeros((len(states), len(actions)), dtype=np.ndarray)
        reward_matrix = np.zeros((len(states), len(actions), len(states)))
        for state in states:
            for action in actions:
                state_idx = self.state_to_idx[state]
                action_idx = self.action_to_idx[action]
                transitions = transitions_dict.get((state, action), None)
                if transitions is None:
                    continue
                transition_probabilities = np.zeros(len(transitions), dtype=object)
                for idx, (successor_state, transition_probability, reward) in enumerate(transitions):
                    successor_state_idx = self.state_to_idx[successor_state]
                    reward_matrix[state_idx, action_idx, successor_state_idx] = reward
                    transition_probabilities[idx] = (successor_state_idx, transition_probability)
                transition_matrix[state_idx, action_idx] = transition_probabilities
        return transition_matrix, reward_matrix

    def get_successors(self, state_idx):
        successor_states = []
        for action_idx in self.idx_to_action.keys():
            transitions = self.transition_matrix[state_idx, action_idx]
            if isinstance(transitions, int):
                continue
            successors_from_action = [successor_state_idx for successor_state_idx, _ in transitions]
            successor_states += successors_from_action
        return successor_states

    def reset(self):
        initial_state_idx = np.random.choice(self.initial_states_idxs)
        return self.idx_to_state[initial_state_idx]

    def step(self, action):
        action_idx = self.action_to_idx[tuple(action)]
        transitions = self.transition_matrix[self.current_state_idx, action_idx]
        if isinstance(transitions, int):
            reward = -1
            current_state = self.idx_to_state[self.current_state_idx]
        else:
            successor_state_idxs, transition_probabilities = zip(*transitions)
            # make sure probabilities sum to one to avoid problems due to rounding errors
            transition_probabilities = np.array(transition_probabilities) / sum(transition_probabilities)
            successor_state_idx = np.random.choice(successor_state_idxs, p=transition_probabilities)
            reward = self.reward_matrix[self.current_state_idx, action_idx, successor_state_idx]
            self.current_state_idx = successor_state_idx
            current_state = self.idx_to_state[successor_state_idx]
        return current_state, reward

    def transform(self, resource_index):
        for state_idx, state in self.idx_to_state.items():
            if state[1][resource_index] == 0:
                continue
            for action_idx, action in self.idx_to_action.items():
                transitions = self.transition_matrix[state_idx, action_idx]
                if isinstance(transitions, int):
                    continue

                def next_state_needs_less_of_resource(state_probability_pair):
                    next_state_idx = state_probability_pair[0]
                    next_state = self.idx_to_state[next_state_idx]
                    return state[1][resource_index] <= next_state[1][resource_index]

                transitions = list(filter(next_state_needs_less_of_resource, transitions))
                next_states, probabilities = list(zip(*transitions))
                probabilities = np.array(probabilities)
                normalised_probabilities = probabilities / probabilities.sum()
                normalised_transitions = list(zip(next_states, normalised_probabilities))
                self.transition_matrix[state_idx, action_idx] = normalised_transitions



class RestrictedMDP:

    def __init__(self, mdp, cc_state_idxs, hull_values):
        self.mdp = mdp
        self.cc_state_idxs = cc_state_idxs
        successors = list(itertools.chain.from_iterable([mdp.get_successors(s) for s in cc_state_idxs]))
        self.hull = list(set(successors) - set(cc_state_idxs))
        self.hull_values = hull_values
        self.current_state_idx = self.mdp.state_to_idx[self.reset()]

    def reset(self):
        initial_state_idx = np.random.choice(self.cc_state_idxs)
        self.current_state_idx = initial_state_idx
        return self.mdp.idx_to_state[initial_state_idx]

    def step(self, action):
        done = False
        if self.current_state_idx in self.hull:
            reward = self.hull_values[self.current_state_idx]
            current_state = self.mdp.idx_to_state[self.current_state_idx]
            done = True
        else:
            action_idx = self.mdp.action_to_idx[tuple(action)]
            transitions = self.mdp.transition_matrix[self.current_state_idx, action_idx]
            if isinstance(transitions, int):
                reward = -1
                current_state = self.mdp.idx_to_state[self.current_state_idx]
            else:
                successor_state_idxs, transition_probabilities = zip(*transitions)
                transition_probabilities = np.array(transition_probabilities) / sum(transition_probabilities)
                successor_state_idx = np.random.choice(successor_state_idxs, p=transition_probabilities)
                reward = self.mdp.reward_matrix[self.current_state_idx, action_idx, successor_state_idx]
                self.current_state_idx = successor_state_idx
                current_state = self.mdp.idx_to_state[successor_state_idx]
        return current_state, reward, done

    def idx_to_state_list(self, state_idx):
        state = self.mdp.idx_to_state[state_idx]
        state_list = reduce(lambda x, y: list(x) + list(y), state)
        return state_list

def bit_permutations(n):
    if n == 1:
        return [np.array([0]), np.array([1])]
    else:
        sequences = []
        shorter_sequences = bit_permutations(n - 1)
        for s in shorter_sequences:
            sequences.append(np.concatenate((s, [0])))
            sequences.append(np.concatenate((s, [1])))
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
            waiting_tasks, running_tasks, _ = state
            waiting_tasks = np.array(waiting_tasks)
            running_tasks = np.array(running_tasks)
            max_available_resources = self.rap.get_max_resource_availabilities()
            for action in actions:
                tasks_after_allocation = running_tasks + action
                resource_requirement_after_allocation = self.rap.calculate_resources_used(tasks_after_allocation)
                if not((waiting_tasks - action >= 0).all()
                       and (resource_requirement_after_allocation <= max_available_resources).all()):
                    continue
                state_action_transitions = []
                reward = np.sum(action * self.rap.get_rewards())
                possible_departing_tasks = self.get_possible_departing_tasks(tasks_after_allocation)
                for departing_tasks in possible_departing_tasks:
                    departing_tasks = np.array(departing_tasks)
                    departure_probability = self.get_departure_probabilities(tasks_after_allocation, departing_tasks)
                    remaining_tasks = tasks_after_allocation - departing_tasks
                    new_resource_requirement = self.rap.calculate_resources_used(remaining_tasks)
                    resources_left = self.rap.get_max_resource_availabilities() - new_resource_requirement
                    for arriving_tasks in actions:
                        successor_state = (tuple(arriving_tasks), tuple(remaining_tasks), tuple(resources_left))
                        arrival_probability = arrival_probabilities[tuple(arriving_tasks)]
                        transition_probability = departure_probability * arrival_probability
                        transition = (successor_state, transition_probability, reward)
                        state_action_transitions.append(transition)
                transitions[(state, tuple(action))] = state_action_transitions

        initial_states = []
        arriving_tasks_root = tuple(0 for _ in range(self.rap.get_task_count()))
        resources_available_root = tuple(r for r in self.rap.get_max_resource_availabilities())

        actions = [tuple(a) for a in actions]
        for action in actions:
            initial_states.append((action, arriving_tasks_root, resources_available_root))

        mdp = MarkovDecisionProcess(states, actions, transitions, initial_states)
        return mdp

    def get_arrival_probabilities(self, arrivals):
        arrival_probs = arrivals * self.rap.get_task_arrival_p()
        no_arrival_probs = (1 - arrivals) * (1 - self.rap.get_task_arrival_p())
        p = np.prod(arrival_probs + no_arrival_probs)
        return p

    def get_departure_probabilities(self, running_tasks, departures):
        p = self.rap.get_task_departure_p()
        remaining_tasks = running_tasks - departures
        departure_p = p ** departures
        remain_p = (1 - p) ** remaining_tasks
        result = np.prod(remain_p * departure_p)
        return result

    def get_possible_departing_tasks(self, running_task):
        tasks_departing = self.possible_departing_tasks.get(tuple(running_task), None)
        if tasks_departing is not None:
            return tasks_departing
        if sum(running_task) == 1:
            tasks_departing = {tuple(running_task), (0,) * len(running_task)}
            self.possible_departing_tasks[tuple(running_task)] = tasks_departing
            return tasks_departing
        if sum(running_task) == 0:
            self.possible_departing_tasks[tuple(running_task)] = {tuple(running_task)}
            return {tuple(running_task)}

        all_tasks = np.copy(running_task)
        departing_task_set = {tuple(all_tasks)}
        for idx in range(self.rap.get_task_count()):
            tasks = np.copy(running_task)
            if running_task[idx] > 0:
                tasks[idx] -= 1
                departing_task_set = departing_task_set.union(self.get_possible_departing_tasks(tasks))

        self.possible_departing_tasks[tuple(running_task)] = departing_task_set
        return departing_task_set

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

