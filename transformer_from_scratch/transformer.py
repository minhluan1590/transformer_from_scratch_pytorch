import torch
from torch import nn
from transformer_from_scratch import Encoder, Decoder


class Transformer(nn.Module):
    """
    Transformer class for the model described in "Attention is All You Need".

    The Transformer consists of an Encoder and a Decoder, each of which is composed of multiple layers of EncoderLayer and DecoderLayer respectively.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        src_pad_idx (int): The index of the source padding token.
        tgt_pad_idx (int): The index of the target padding token.
        num_layers (int): The number of EncoderLayer and DecoderLayer layers.
        d_model (int): The dimensionality of the input embeddings.
        heads (int): The number of attention heads.
        dropout (float or None): The dropout probability of the feed-forward network. Defaults to None.
        forward_expansion (int): The forward expansion factor of the feed-forward network.
        max_len (int): The maximum length of the input sequence.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        num_layers: int = 6,
        d_model: int = 512,
        heads: int = 8,
        dropout: float or None = None,
        forward_expansion: int = 4,
        max_len: int = 512,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_layers,
            heads,
            dropout,
            forward_expansion,
            max_len,
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_layers,
            heads,
            dropout,
            forward_expansion,
            max_len,
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        src_mask = src_mask.float().masked_fill(src_mask == 1, float("-inf"))
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        N, tgt_len = tgt.shape
        # The tgt shape is (N, tgt_len), so we need to add two dimensions to the mask to be (N, 1, tgt_len, tgt_len)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )

        # Fill the upper triangle of the mask with -inf, and change the lower triangle to 0
        tgt_mask = (
            tgt_mask.float()
            .masked_fill(tgt_mask == 0, float("-inf"))
            .masked_fill(tgt_mask == 1, 0.0)
        )

        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_src, src_mask, tgt_mask)
