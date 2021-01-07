import torch


def copy_mlp_weights(baselines_model, torch_mlp):
    model_params = baselines_model.get_parameters()

    policy_keys = [key for key in model_params.keys() if "pi" in key]
    policy_params = [model_params[key] for key in policy_keys]

    for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
        param = torch.from_numpy(policy_param)
        # Copies parameters from baselines model to pytorch model
        print(th_key, key)
        print(pytorch_param.shape, param.shape, policy_param.shape)
        pytorch_param.data.copy_(param.data.clone().t())

    return torch_mlp


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

    def __init__(self, n_inputs, n_outputs):
        super(SubspaceMLP, self).__init__()
        sizes = [n_inputs] + [64, 64] + [n_outputs]
        activations = [torch.nn.Tanh(), torch.nn.Tanh(), torch.nn.Softmax(dim=0)]
        self.ff_stream = mlp(sizes, activations)

    def forward(self, x):
        return self.ff_stream(x)


class MultiStageModel(torch.nn.Module):

    def __init__(self, baseline_models, subspace_in, subspace_out):
        super(MultiStageModel, self).__init__()
        self.subspace_mlps = []
        for baseline_model in baseline_models:
            subspace_mlp = SubspaceMLP(subspace_in, subspace_out)
            copy_mlp_weights(baseline_model, subspace_mlp)
            self.subspace_mlps.append(subspace_mlp)

        combining_in = len(self.subspace_mlps) * subspace_out
        sizes = [combining_in] + [64, 64] + [subspace_out]
        activations = [torch.nn.Tanh(), torch.nn.Tanh(), torch.nn.Softmax(dim=0)]
        self.combining_nn = mlp(sizes, activations)

    def forward(self, x):
        stage_1_results = [subspace_mlp.forward(x) for subspace_mlp in self.subspace_mlps]
        stage_1_results = torch.cat(stage_1_results)
        stage_2_result = self.combining_nn(stage_1_results)
        return stage_2_result
