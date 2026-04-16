import torch.nn as nn


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(name)


def build_mlp(input_dim, output_dim, hidden_dims, activation):

    activation_cls = get_activation(activation)

    layers = []
    prev = input_dim

    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation_cls())
        prev = h

    layers.append(nn.Linear(prev, output_dim))

    return nn.Sequential(*layers)