from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GRUInputs:
    input_size: int
    hidden_size: int
    sigma: float

class GRU(nn.Module):
    def __init__(self, inputs: GRUInputs):
        super().__init__()
        self.input_size = inputs.input_size
        self.hidden_size = inputs.hidden_size
        self.W = nn.Parameter(torch.randn(inputs.input_size, 3 * inputs.hidden_size) * inputs.sigma)
        self.W_h = nn.Parameter(torch.randn(inputs.hidden_size, 3 * inputs.hidden_size) * inputs.sigma)
        self.b = nn.Parameter(torch.zeros(3 * inputs.hidden_size))
        
    def forward(self, X: torch.Tensor, H: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = X.shape
        
        if H is None:
            H = X.new_zeros((batch_size, self.hidden_size))
        
        outputs = []
        for t in range(sequence_length):
            x_t = X[:, t, :]
            x_z, x_r, x_n = (x_t @ self.W + self.b).chunk(3, dim=1)
            h_z, h_r, h_n = (H @ self.W_h).chunk(3, dim=1)
            z_t = torch.sigmoid(x_z + h_z)
            r_t = torch.sigmoid(x_r + h_r)
            n_t = torch.tanh(x_n + r_t * h_n)
            H = (1 - z_t) * n_t + z_t * H
            outputs.append(H)

        return torch.stack(outputs, dim=1), H
    
class GRUMultiLayer(nn.Module):
    def __init__(self, layers: list[GRU]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.validate_layers()

    def validate_layers(self):
        for previous, current in zip(self.layers, self.layers[1:]):
            if previous.hidden_size != current.input_size:
                raise ValueError(
                    f"Hidden size of previous layer ({previous.hidden_size}) "
                    f"must match input size of next layer ({current.input_size})"
                )

    def forward(
        self,
        X: torch.Tensor,
        states: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if states is None:
            states = [None] * len(self.layers)
        elif len(states) != len(self.layers):
            raise ValueError("One state is required for each GRU layer")
        
        new_states = []
        for layer, state in zip(self.layers, states):
            X, new_state = layer(X, state)
            new_states.append(new_state)
        
        return X, new_states