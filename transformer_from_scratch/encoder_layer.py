import torch
from torch import nn
from transformer_from_scratch import MultiHeadAttention, AddNorm, FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Transformer Block class that implements a single block of the transformer architecture.

    This class combines a multi-head self-attention mechanism, two add & norm (layer normalization) steps,
    and a position-wise feed-forward network.

    Attributes:
        attention (MultiHeadAttention): The multi-head self-attention mechanism.
        norm1 (AddNorm): The first add & norm (layer normalization) step.
        ffn (FeedForwardNetwork): The position-wise feed-forward network.
        norm2 (AddNorm): The second add & norm (layer normalization) step.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        heads (int): The number of attention heads.

    Methods:
        forward(value: Tensor, key: Tensor, query: Tensor, mask: Tensor) -> Tensor:
            Passes the input through the transformer block and returns the output.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float or None = None,
        forward_expansion: int = 4,
    ):
        """
        The constructor of the EncoderLayer class.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            heads (int): The number of attention heads.
            dropout (float or None): The dropout probability of the feed-forward network. Defaults to None.
            forward_expansion (int): The forward expansion factor of the feed-forward network.
        """
        super(EncoderLayer, self).__init__()

        # Initialize the multi-head self-attention mechanism
        self.attention = MultiHeadAttention(d_model, heads)

        # Initialize the first add & norm (layer normalization) step
        self.norm1 = AddNorm(d_model)

        # Initialize the position-wise feed-forward network
        self.ffn = FeedForwardNetwork(d_model, dropout, forward_expansion)

        # Initialize the second add & norm (layer normalization) step
        self.norm2 = AddNorm(d_model)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pass the input through the multi-head self-attention mechanism
        attention = self.attention(value, key, query, mask)

        # Pass the attention output through the first add & norm (layer normalization) step
        x = self.norm1(query + attention)

        # Pass the output through the position-wise feed-forward network
        ffn = self.ffn(x)

        return self.norm2(x + ffn)
