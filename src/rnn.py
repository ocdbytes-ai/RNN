from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from torch import nn
from typing_extensions import override


@dataclass
class RNNInputs:
    embedding_size: int
    hidden_layer_size: int
    sigma: float

class RNN(nn.Module):
    def __init__(self, rnn_inputs: RNNInputs):
        super().__init__()
        self.rnn_inputs: RNNInputs = rnn_inputs
        self.W: nn.Parameter = nn.Parameter(
            torch.randn(rnn_inputs.embedding_size, rnn_inputs.hidden_layer_size) * rnn_inputs.sigma
        )
        self.W_h: nn.Parameter = nn.Parameter(
            torch.randn(rnn_inputs.hidden_layer_size, rnn_inputs.hidden_layer_size) * rnn_inputs.sigma
        )
        self.b: nn.Parameter = nn.Parameter(
            torch.zeros(rnn_inputs.hidden_layer_size)
        )

    @override
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
        outputs: list[torch.Tensor] = []
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
           self.validate_layers(rnns)
           self.layers: nn.ModuleList = nn.ModuleList(rnns)

    @staticmethod
    def validate_layers(rnns: list[RNN]) -> None:
           for previous, current in pairwise(rnns):
               if (
                   current.rnn_inputs.embedding_size
                   != previous.rnn_inputs.hidden_layer_size
               ):
                   raise ValueError(
                       "Each layer's embedding_size must equal the previous layer's hidden_layer_size"
                   )

    @override
    def forward(
           self,
           x: torch.Tensor,
           states: list[torch.Tensor | None] | None = None,
       ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            # x: (batch_size, sequence_length, embedding_size)
            layer_states: list[torch.Tensor | None] = (
                states if states is not None else [None] * len(self.layers)
            )
            if len(layer_states) != len(self.layers):
               raise ValueError("One state is required for each RNN layer")

            # final_states : (num_layers, batch_size, hidden_layer_size[layer_i])
            final_states: list[torch.Tensor] = []

            # Iter over each layer with corresponding state
            # the input to the next layer is the states
            # collected for each sequence step
            for layer, state in zip(cast(Iterable[RNN], self.layers), layer_states):
               x, state = cast(tuple[torch.Tensor, torch.Tensor], layer(x, state))
               final_states.append(state)

            # final_states is the last state calculated in the last
            # data sequence
            # shape :
            # (batch_size, sequence_length, last_layer.hidden_layer_size),
            # (num_layers, batch_size, hidden_layer_size[layer_i])
            return x, final_states