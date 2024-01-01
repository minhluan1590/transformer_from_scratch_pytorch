import torch
from torch import nn

from transformer_from_scratch import EncoderLayer, PositionalEncoding


class Encoder(nn.Module):
    """
    Encoder class for Transformer in "Attention is All You Need".

    The Encoder consists of multiple layers of EncoderLayer, each of which
    contains a MultiHeadAttention layer and a FeedForward layer, both followed
    by AddNorm (add & normalization) layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the input embeddings.
        num_layers (int): The number of EncoderLayer layers.
        heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        forward_expansion (int): The forward expansion factor of the feed-forward network.
        max_len (int): The maximum length of the input sequence.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        heads,
        dropout,
        forward_expansion,
        max_len,
    ):
        super(Encoder, self).__init__()

        # Initialize the EncoderLayer layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding Layer
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x, mask):
        """
        Passes the input through the Encoder.

        Args:
            x (Tensor): The input to the Encoder.
            mask (Tensor): The mask to apply to the input.

        Returns:
            The output of the Encoder.
        """

        # Get the batch size and sequence length
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Pass the input through the word embedding layer
        out = self.embedding(x)

        # Pass the embeddings through the positional encoding layer
        out = self.positional_encoding(out)

        # Pass the output through the EncoderLayer layers
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
