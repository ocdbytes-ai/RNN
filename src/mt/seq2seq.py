from typing import cast

import torch
from torch import nn
from typing_extensions import override

EncoderState = tuple[torch.Tensor, torch.Tensor]
DecoderState = tuple[torch.Tensor, torch.Tensor]


class Encoder(nn.Module):
    @override
    def forward(
        self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None
    ) -> EncoderState:
        raise NotImplementedError


class Decoder(nn.Module):
    def init_state(self, _encoded: EncoderState) -> DecoderState:
        raise NotImplementedError

    @override
    def forward(
        self, x: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState]:
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    @override
    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        source_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = cast(EncoderState, self.encoder(encoder_input, source_lengths))
        state = self.decoder.init_state(encoded)
        output, _ = cast(
            tuple[torch.Tensor, DecoderState], self.decoder(decoder_input, state)
        )
        return output
