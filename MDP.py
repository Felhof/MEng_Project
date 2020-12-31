import numpy as np


def build_index(items):
    item_to_idx = {}
    idx_to_item = {}

    for idx, item in enumerate(items):
        item_to_idx[item] = idx
        idx_to_item[idx] = item

    return item_to_idx, idx_to_item


class MarkovDecisionProcess:

    def __init__(self, states, actions, transitions):
        self.state_to_idx, self.idx_to_state = build_index(states)
        self.action_to_idx, self.idx_to_action = build_index(actions)
        self.transition_matrix = self.build_transition_matrix(states, actions, transitions)

    def build_transition_matrix(self, states, actions, transitions):
        transition_matrix = np.zeros((len(states), len(actions)), dtype=np.ndarray)
        for state in states:
            for action in actions:
                state_idx = self.state_to_idx[state]
                action_idx = self.action_to_idx[action]
                transition_matrix[state_idx, action_idx] = transitions[(state, action)]
        return transition_matrix

    def get_successors(self, state):
        state_idx = self.state_to_idx[state]
        successor_states = []
        for action_idx in self.idx_to_action.keys():
            successors_from_action = [successor_state for successor_state, _, _ in self.transition_matrix[state_idx, action_idx]]
            successor_states += successors_from_action
        return successor_states
