import time
import numpy as np
import torch


class SACAgent:

    ALPHA = 0.1
    BATCH_SIZE = 100
    DISCOUNT_RATE = 0.9
    NUM_ACTIONS = 4

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self):
        self.critic_local = Network(input_dimension=2, output_dimension=4)
        self.critic_local2 = Network(input_dimension=2, output_dimension=4)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=0.001)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=0.001)

        self.critic_target = Network(input_dimension=2, output_dimension=4)
        self.critic_target2 = Network(input_dimension=2, output_dimension=4)

        self.update_target_networks()

        self.actor_local = Network(
            input_dimension=2, output_dimension=4, output_activation=torch.nn.Softmax(dim=1)
        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer()

    def get_next_action(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.random.choice(range(self.NUM_ACTIONS), p=action_probabilities)
        return discrete_action

    def train_on_transition(self, state, discrete_action, next_state, reward):
        transition = (state, discrete_action, reward, next_state)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(minibatch_separated[0])
            actions_tensor = torch.tensor(minibatch_separated[1])
            rewards_tensor = torch.tensor(minibatch_separated[2]).float()
            next_states_tensor = torch.tensor(minibatch_separated[3])

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.ALPHA * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + self.DISCOUNT_RATE*soft_state_values

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.ALPHA * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def predict(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def discrete_to_continuous_action(self, discrete_action):
        return {
            self.UP: np.array([0, 0.01], dtype=np.float32),
            self.RIGHT: np.array([0.01, 0], dtype=np.float32),
            self.DOWN: np.array([0, -0.01], dtype=np.float32),
            self.LEFT: np.array([-0.01, 0], dtype=np.float32)
        }[discrete_action]

    def update_target_networks(self):
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output


class ReplayBuffer:

    def __init__(self, capacity=5000):
        self.buffer = np.zeros(capacity, dtype='2float32, i8, float32, 2float32')
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count
