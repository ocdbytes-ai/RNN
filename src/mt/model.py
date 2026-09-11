from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing_extensions import override

from .seq2seq import Decoder, DecoderState, Encoder, EncoderDecoder, EncoderState


@dataclass
class Seq2SeqInputs:
    vocab_size: int
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.0


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        _ = nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                _ = nn.init.xavier_uniform_(parameter)


class Seq2SeqEncoder(Encoder):
    def __init__(self, inputs: Seq2SeqInputs):
        super().__init__()
        self.inputs: Seq2SeqInputs = inputs
        self.embedding: nn.Embedding = nn.Embedding(
            inputs.vocab_size, inputs.embedding_size
        )
        self.gru: nn.GRU = nn.GRU(
            inputs.embedding_size,
            inputs.hidden_size,
            num_layers=inputs.num_layers,
            dropout=inputs.dropout,
            batch_first=True,
        )
        _ = self.apply(init_weights)

    @override
    def forward(
        self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None
    ) -> EncoderState:
        embeddings = cast(torch.Tensor, self.embedding(x.long()))
        if valid_lengths is None:
            return cast(EncoderState, self.gru(embeddings))

        # Here I am using packed padded sequence
        # because I dont want my model to consider
        # the padded tokens' context.
        packed = pack_padded_sequence(
            embeddings,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, state = cast(
            tuple[PackedSequence, torch.Tensor], self.gru(packed)
        )
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=x.shape[1],
        )
        return outputs, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, inputs: Seq2SeqInputs):
        super().__init__()
        self.inputs: Seq2SeqInputs = inputs
        self.embedding: nn.Embedding = nn.Embedding(
            inputs.vocab_size, inputs.embedding_size
        )
        self.gru: nn.GRU = nn.GRU(
            inputs.embedding_size + inputs.hidden_size,
            inputs.hidden_size,
            num_layers=inputs.num_layers,
            dropout=inputs.dropout,
            batch_first=True,
        )
        self.output: nn.Linear = nn.Linear(inputs.hidden_size, inputs.vocab_size)
        _ = self.apply(init_weights)

    @override
    def init_state(self, encoded: EncoderState) -> DecoderState:
        _, hidden_state = encoded
        return hidden_state[-1], hidden_state

    @override
    def forward(
        self, x: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState]:
        embeddings = cast(torch.Tensor, self.embedding(x.long()))
        # from encoder
        context, hidden_state = state
        # context shape : (batch_size, hidden_size)
        # Copy context to all the embeddings for each time stamp
        expanded_context = context.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        # concatenate embeddings and context
        # shape : (batch_size, target_steps, embedding_size + hidden_size)
        gru_input = torch.cat((embeddings, expanded_context), dim=-1)
        outputs, hidden_state = cast(
            tuple[torch.Tensor, torch.Tensor], self.gru(gru_input, hidden_state)
        )
        # batch_first=True already provides batch based representation.
        logits = cast(torch.Tensor, self.output(outputs))
        # logits shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, hidden_size)
        return logits, (context, hidden_state)


class Seq2Seq(EncoderDecoder):
    pass
