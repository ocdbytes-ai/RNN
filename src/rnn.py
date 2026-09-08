from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RNNInputs:
    embedding_size: int
    hidden_layer_size: int
    sigma: float

class RNN(nn.Module):
    def __init__(self, rnn_inputs: RNNInputs):
        super().__init__()
        self.rnn_inputs = rnn_inputs
        self.W = nn.Parameter(
            torch.randn(rnn_inputs.embedding_size, rnn_inputs.hidden_layer_size) * rnn_inputs.sigma
        )
        self.W_h = nn.Parameter(
            torch.randn(rnn_inputs.hidden_layer_size, rnn_inputs.hidden_layer_size) * rnn_inputs.sigma
        )
        self.b = nn.Parameter(
            torch.zeros(rnn_inputs.hidden_layer_size)
        )

    def forward(
        self,
        X: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # X: (batch_size, sequence_length, embedding_size)
        if state is None:
            # separate state for each example in the batch
            # state : (batch_size, hidden_layer_size)
            state = X.new_zeros(
                (X.shape[0], self.rnn_inputs.hidden_layer_size)
            )

        # outputs : (sequence_length, batch_size, hidden_layer_size)
        outputs = []
        # For looping over the sequences (dimension 1) because of the X shape (batch_size, sequence_length, embedding_size)
        for x in X.unbind(dim=1):
            # x: (batch_size, embedding_size)
            # state : (batch_size, hidden_layer_size)
            state = torch.tanh(
                x @ self.W + state @ self.W_h + self.b
            )
            outputs.append(state)
        # last state is the output of the layer returned as `state`
        # each batch has it's own state.
        # `stack`` creates a new dimension and puts the tensors along that dimension.
        # shape : (batch_size, sequence_length, hidden_layer_size), (batch_size, hidden_layer_size)
        return torch.stack(outputs, dim=1), state

class RNNMultiLayer(nn.Module):
    def __init__(self, rnns: list[RNN]):
           super().__init__()
           self.layers = nn.ModuleList(rnns)
           self.validate_layers()

    def validate_layers(self) -> None:
           for previous, current in zip(self.layers, self.layers[1:]):
               if (
                   current.rnn_inputs.embedding_size
                   != previous.rnn_inputs.hidden_layer_size
               ):
                   raise ValueError(
                       "Each layer's embedding_size must equal the previous "
                       "layer's hidden_layer_size"
                   )

    def forward(
           self,
           X: torch.Tensor,
           states: list[torch.Tensor | None] | None = None,
       ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            # X: (batch_size, sequence_length, embedding_size)
            if states is None:
               states = [None] * len(self.layers)
            elif len(states) != len(self.layers):
               raise ValueError("One state is required for each RNN layer")

            # final_states : (num_layers, batch_size, hidden_layer_size[layer_i])
            final_states = []

            # Iter over each layer with corresponding state
            # the input to the next layer is the states 
            # collected for each sequence step
            for layer, state in zip(self.layers, states):
               X, state = layer(X, state)
               final_states.append(state)

            # final_states is the last state calculated in the last
            # data sequence
            # shape : 
            # (batch_size, sequence_length, last_layer.hidden_layer_size), 
            # (num_layers, batch_size, hidden_layer_size[layer_i])
            return X, final_states