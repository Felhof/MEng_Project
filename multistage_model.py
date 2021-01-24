import torch
from stable_baselines3.common.policies import ActorCriticPolicy

import copy


def mlp(sizes, activations):
    layers = []
    connections = [
        (in_dim, out_dim) for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
    ]
    for connection, activation in zip(connections, activations):
        layers.append(torch.nn.Linear(connection[0], connection[1]))
        layers.append(activation)

    return torch.nn.Sequential(*layers)


class SubspaceMLP(torch.nn.Module):

    def __init__(self, baseline_model):
        super(SubspaceMLP, self).__init__()
        self.mlp_extractor = copy.deepcopy(baseline_model.policy.mlp_extractor)
        self.action_net = copy.deepcopy(baseline_model.policy.action_net)
        self.value_net = copy.deepcopy(baseline_model.policy.value_net)

        self.softmax = torch.nn.Softmax()

    def forward(self, observation):
        latent_pi, latent_vf = self.mlp_extractor(observation)
        action_logits = self.action_net(latent_pi)
        action_probs = self.softmax(action_logits)
        values = self.value_net(latent_vf)
        return action_probs, values


class MultiStageNetwork(torch.nn.Module):

    def __init__(self, stage1_models, action_dim):
        super(MultiStageNetwork, self).__init__()
        self.stage1_mlps = []
        for stage1_model in stage1_models:
            subspace_mlp = SubspaceMLP(stage1_model)
            self.subspace_mlps.append(subspace_mlp)

        pi_sizes = [action_dim * len(self.stage1_mlps), 64, 64]
        vf_sizes = [len(self.stage1_mlps), 64, 64]
        activations = [torch.nn.Tanh(), torch.nn.Tanh()]

        self.stage2_policy_net = mlp(pi_sizes, activations)
        self.stage2_value_net = mlp(vf_sizes, activations)

    def forward(self, observation):
        stage1_pi_outputs = []
        stage1_vf_outputs = []
        for stage1_mlp in self.stage1_mlps:
            action_probs, values = stage1_mlp(observation)
            stage1_pi_outputs.append(action_probs)
            stage1_vf_outputs.append(values)
        aggregate_pi = torch.cat(stage1_pi_outputs)
        aggregate_vf = torch.cat(stage1_vf_outputs)
        latent_pi = self.stage2_policy_net(aggregate_pi)
        latent_vf = self.stage2_value_net(aggregate_vf)
        return latent_pi, latent_vf


class MultiStageActorCritic(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=torch.nn.Tanh,
        *args,
        **kwargs,
    ):
        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.stage1_models = kwargs["stage1_models"]
        self.actions = action_space.n

    def _build_mlp_extractor(self):
        self.mlp_extractor = MultiStageNetwork(self.stage1_models, self.actions)
