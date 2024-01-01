import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the paper "Attention is All You Need".
    Positional encoding is used in the Transformer model to give the model some information about
    the relative position of the words in the sentence.

    Args:
        d_model (int): The dimension of the embeddings.
        max_len (int, optional): The maximum length of the sequences. Defaults to 5000.

    Attributes:
        pe (torch.Tensor): The positional encodings for the sequences.

    Methods:
        forward(x): Adds the positional encodings to the input embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
