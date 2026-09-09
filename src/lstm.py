from dataclasses import dataclass

import torch
from torch import nn


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
        self.input_size = inputs.input_size
        self.hidden_size = inputs.hidden_size
        gates_size = 4 * inputs.hidden_size
        self.W = nn.Parameter(torch.randn(inputs.input_size, gates_size) * inputs.sigma)
        self.W_h = nn.Parameter(torch.randn(inputs.hidden_size, gates_size) * inputs.sigma)
        self.b = nn.Parameter(torch.zeros(gates_size))

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
            C = X.new_zeros((batch_size, self.hidden_size))
            H = X.new_zeros((batch_size, self.hidden_size))
        else:
            C, H = states.C, states.H

        # Hidden states for the sequences
        outputs = []
        
        for t in range(sequence_length):
            x_t = X[:, t, :]  # Get the input at time step t

            # Unoptimized equivalent with separate parameters:
            # i_t = torch.sigmoid(x_t @ self.W_i + H @ self.W_h_i + self.b_i)
            # f_t = torch.sigmoid(x_t @ self.W_f + H @ self.W_h_f + self.b_f)
            # g_t = torch.tanh(x_t @ self.W_c + H @ self.W_h_c + self.b_c)
            # o_t = torch.sigmoid(x_t @ self.W_o + H @ self.W_h_o + self.b_o)

            # Compute all four gates with two matrix multiplications.
            i_t, f_t, g_t, o_t = (x_t @ self.W + H @ self.W_h + self.b).chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Update cell state
            C = f_t * C + i_t * g_t
            # Update hidden state
            H = o_t * torch.tanh(C)

            outputs.append(H)

        # This is for the corpus format I am using
        output_tensor = torch.stack(outputs, dim=1)  # Stack outputs along the sequence dimension
        new_states = LSTMStates(C=C, H=H)  # Create new states object

        return output_tensor, new_states

class LSTMMultiLayer(nn.Module):
    def __init__(self, lstms: list[LSTM]):
        super().__init__()
        self.layers = nn.ModuleList(lstms)
        self.validate_layers()

    def validate_layers(self) -> None:
        for previous, current in zip(self.layers, self.layers[1:]):
            if current.input_size != previous.hidden_size:
                raise ValueError(
                    "Each layer's input_size must equal the previous "
                    "layer's hidden_size"
                )

    def forward(
        self,
        X: torch.Tensor,
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
        if states is None:
            states = [None] * len(self.layers)
        elif len(states) != len(self.layers):
            raise ValueError("One state is required for each LSTM layer")

        new_states = []
        for layer, state in zip(self.layers, states):
            X, new_state = layer(X, state)
            new_states.append(new_state)

        return X, new_states