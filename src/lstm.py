from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from torch import nn
from typing_extensions import override


@dataclass
class LSTMInputs:
    """
    Represents the input parameters for an LSTM cell.
    
    input_size: The number of input features (can be embedding size etc.)
    hidden_size: The number of hidden features
    sigma: The standard deviation for weight initialisation
    """

    input_size: int
    hidden_size: int
    sigma: float

@dataclass
class LSTMStates:
    """
    Represents the states of an LSTM cell.
    
    C: internal cell state (memory) of the LSTM
    H: hidden state (output) of the LSTM
    """

    C: torch.Tensor
    H: torch.Tensor

class LSTM(nn.Module):
    def __init__(self, inputs: LSTMInputs):
        super().__init__()
        self.input_size: int = inputs.input_size
        self.hidden_size: int = inputs.hidden_size
        gates_size = 4 * inputs.hidden_size
        self.W: nn.Parameter = nn.Parameter(torch.randn(inputs.input_size, gates_size) * inputs.sigma)
        self.W_h: nn.Parameter = nn.Parameter(torch.randn(inputs.hidden_size, gates_size) * inputs.sigma)
        self.b: nn.Parameter = nn.Parameter(torch.zeros(gates_size))

    @override
    def forward(self, X: torch.Tensor, states: LSTMStates | None = None) -> tuple[torch.Tensor, LSTMStates]:
        """
        Forward pass through the LSTM cell.

        Args:
            X: Input tensor of shape (batch_size, sequence_length, input_size)
            states: LSTMStates containing the previous cell state and hidden state

        Returns:
            output: Output tensor of shape (batch_size, sequence_length, hidden_size)
            new_states: LSTMStates containing the updated cell state and hidden state
        """
        batch_size, sequence_length, _ = X.shape

        if states is None:
            # Initialize cell state and hidden state to zeros if not provided
            c = X.new_zeros((batch_size, self.hidden_size))
            h = X.new_zeros((batch_size, self.hidden_size))
        else:
            c, h = states.C, states.H

        # Hidden states for the sequences
        outputs: list[torch.Tensor] = []
        
        for t in range(sequence_length):
            x_t = X[:, t, :]  # Get the input at time step t

            # Unoptimized equivalent with separate parameters:
            # i_t = torch.sigmoid(x_t @ self.W_i + H @ self.W_h_i + self.b_i)
            # f_t = torch.sigmoid(x_t @ self.W_f + H @ self.W_h_f + self.b_f)
            # g_t = torch.tanh(x_t @ self.W_c + H @ self.W_h_c + self.b_c)
            # o_t = torch.sigmoid(x_t @ self.W_o + H @ self.W_h_o + self.b_o)

            # Compute all four gates with two matrix multiplications.
            i_t, f_t, g_t, o_t = (x_t @ self.W + h @ self.W_h + self.b).chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Update cell state
            c = f_t * c + i_t * g_t
            # Update hidden state
            h = o_t * torch.tanh(c)

            outputs.append(h)

        # This is for the corpus format I am using
        output_tensor = torch.stack(outputs, dim=1)  # Stack outputs along the sequence dimension
        new_states = LSTMStates(C=c, H=h)  # Create new states object

        return output_tensor, new_states

class LSTMMultiLayer(nn.Module):
    def __init__(self, lstms: list[LSTM]):
        super().__init__()
        self.validate_layers(lstms)
        self.layers: nn.ModuleList = nn.ModuleList(lstms)

    @staticmethod
    def validate_layers(lstms: list[LSTM]) -> None:
        for previous, current in pairwise(lstms):
            if current.input_size != previous.hidden_size:
                raise ValueError(
                    "Each layer's input_size must equal the previous layer's hidden_size"
                )

    @override
    def forward(
        self,
        x: torch.Tensor,
        states: list[LSTMStates | None] | None = None,
    ) -> tuple[torch.Tensor, list[LSTMStates]]:
        """
        Forward pass through the multi-layer LSTM.

        Args:
            X: Input tensor of shape (batch_size, sequence_length, input_size)
            states: List of LSTMStates for each layer or None

        Returns:
            output: Output tensor of shape (batch_size, sequence_length, hidden_size[layer_last])
            new_states: List of updated LSTMStates for each layer
        """
        layer_states: list[LSTMStates | None] = (
            states if states is not None else [None] * len(self.layers)
        )
        if len(layer_states) != len(self.layers):
            raise ValueError("One state is required for each LSTM layer")

        new_states: list[LSTMStates] = []
        for layer, state in zip(cast(Iterable[LSTM], self.layers), layer_states):
            x, new_state = cast(tuple[torch.Tensor, LSTMStates], layer(x, state))
            new_states.append(new_state)

        return x, new_states