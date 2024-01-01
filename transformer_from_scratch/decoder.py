import torch
from torch import nn

from transformer_from_scratch import PositionalEncoding, DecoderLayer


class Decoder(nn.Module):
    """
    Decoder class for Transformer in "Attention is All You Need".

    The Decoder consists of multiple layers of DecoderLayer, each of which
    contains two MultiHeadAttention layers (one for self-attention, one for encoder-decoder attention)
    and a FeedForward layer, all followed by AddNorm (add & normalization) layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the input embeddings.
        num_layers (int): The number of DecoderLayer layers.
        heads (int): The number of attention heads.
        dropout (float or None): The dropout probability of the feed-forward network. Defaults to None.
        forward_expansion (int): The forward expansion factor of the feed-forward network.
        max_len (int): The maximum length of the input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        heads: int,
        dropout: float or None = None,
        forward_expansion: int = 4,
        max_len: int = 512,
    ):
        super(Decoder, self).__init__()

        # Initialize the word embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Initialize the positional encoding layer
        self.position_encoding = PositionalEncoding(d_model, max_len)

        # Initialize the dropout layer
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, heads, dropout, forward_expansion))

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self, x, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ):
        # Calculate the word embeddings
        x = self.embedding(x)

        # Pass the word embeddings through the positional encoding layer
        x = self.position_encoding(x)

        # Apply the dropout if specified
        if self.dropout:
            x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        return self.fc_out(x)
