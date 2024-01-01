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
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, heads, dropout, forward_expansion))

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_encoding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        out = self.fc_out(x)

        return out
