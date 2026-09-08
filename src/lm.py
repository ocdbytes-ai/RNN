from dataclasses import dataclass

import torch
from torch import nn

from .rnn import RNN, RNNInputs, RNNMultiLayer


@dataclass
class LanguageModelParams:
    """
    Parameters for the language model.
    
    Size of the total vocabulary in the corpus
    vocab_size: int
    
    Size of the input embedding
    embedding_size: int
    """
    
    vocab_size: int
    embedding_size: int

class LM(nn.Module):
    def __init__(self, lm_params: LanguageModelParams, rnn_params: list[RNNInputs]):
        """
        lm_params: Parameters for the language model
        rnn_params: List of RNNInputs for each layer of the RNN
        """
        super().__init__()
        self.lm_params = lm_params
        self.rnn_params = rnn_params
        self.embedding = nn.Embedding(
            lm_params.vocab_size,
            lm_params.embedding_size,
        )
        self.rnn = RNNMultiLayer([RNN(rnn_input) for rnn_input in rnn_params])
        self.output_layer = nn.Linear(rnn_params[-1].hidden_layer_size, lm_params.vocab_size)

    def forward(
        self,
        X: torch.Tensor,
        states: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the language model.

        X: (batch_size, sequence_length)
        states: (num_layers, batch_size, hidden_layer_size[layer_i]) or None
        
        Returns:
            output: (batch_size, sequence_length, vocab_size)
            state: (num_layers, batch_size, hidden_layer_size[layer_i])
        """
        # X: (batch_size, sequence_length, embedding_size)
        X = self.embedding(X)
        X, states = self.rnn(X, states)
        # output: (batch_size, sequence_length, vocab_size)
        # state: (num_layers, batch_size, hidden_layer_size[layer_i])
        output = self.output_layer(X)
        return output, states