from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from torch import nn
from typing_extensions import override


@dataclass
class GRUInputs:
    input_size: int
    hidden_size: int
    sigma: float

class GRU(nn.Module):
    def __init__(self, inputs: GRUInputs):
        super().__init__()
        self.input_size: int = inputs.input_size
        self.hidden_size: int = inputs.hidden_size
        self.W: nn.Parameter = nn.Parameter(torch.randn(inputs.input_size, 3 * inputs.hidden_size) * inputs.sigma)
        self.W_h: nn.Parameter = nn.Parameter(torch.randn(inputs.hidden_size, 3 * inputs.hidden_size) * inputs.sigma)
        self.b: nn.Parameter = nn.Parameter(torch.zeros(3 * inputs.hidden_size))

    @override
    def forward(self, X: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = X.shape

        if h is None:
            h = X.new_zeros((batch_size, self.hidden_size))

        outputs: list[torch.Tensor] = []
        for t in range(sequence_length):
            x_t = X[:, t, :]
            x_z, x_r, x_n = (x_t @ self.W + self.b).chunk(3, dim=1)
            h_z, h_r, h_n = (h @ self.W_h).chunk(3, dim=1)
            z_t = torch.sigmoid(x_z + h_z)
            r_t = torch.sigmoid(x_r + h_r)
            n_t = torch.tanh(x_n + r_t * h_n)
            h = (1 - z_t) * n_t + z_t * h
            outputs.append(h)

        return torch.stack(outputs, dim=1), h
    
class GRUMultiLayer(nn.Module):
    def __init__(self, layers: list[GRU]):
        super().__init__()
        self.validate_layers(layers)
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @staticmethod
    def validate_layers(layers: list[GRU]) -> None:
        for previous, current in pairwise(layers):
            if previous.hidden_size != current.input_size:
                raise ValueError(
                    f"Hidden size of previous layer ({previous.hidden_size}) must match input size of next layer ({current.input_size})"
                )

    @override
    def forward(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layer_states: list[torch.Tensor | None] = (
            states if states is not None else [None] * len(self.layers)
        )
        if len(layer_states) != len(self.layers):
            raise ValueError("One state is required for each GRU layer")

        new_states: list[torch.Tensor] = []
        for layer, state in zip(cast(Iterable[GRU], self.layers), layer_states):
            x, new_state = cast(tuple[torch.Tensor, torch.Tensor], layer(x, state))
            new_states.append(new_state)

        return x, new_states