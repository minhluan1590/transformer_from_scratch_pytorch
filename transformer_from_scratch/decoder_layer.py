from torch import nn
from transformer_from_scratch import MultiHeadAttention, AddNorm, FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    DecoderLayer class that implements a single block of the decoder part of the transformer architecture.

    This class combines two multi-head attention mechanisms (one for self-attention, one for encoder-decoder attention),
    two add & norm (layer normalization) steps, and a position-wise feed-forward network.

    Attributes:
        self_attention (MultiHeadAttention): The multi-head self-attention mechanism.
        encoder_decoder_attention (MultiHeadAttention): The multi-head encoder-decoder attention mechanism.
        norm1 (AddNorm): The first add & norm (layer normalization) step.
        ffn (FeedForwardNetwork): The position-wise feed-forward network.
        norm2 (AddNorm): The second add & norm (layer normalization) step.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        heads (int): The number of attention heads.

    Methods:
        forward(value: Tensor, key: Tensor, query: Tensor, mask: Tensor) -> Tensor:
            Passes the input through the decoder block and returns the output.
    """

    def __init__(
        self, d_model, heads, dropout: float or None = None, forward_expansion=4
    ):
        super(DecoderLayer, self).__init__()
        # Initialize the multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, heads)
        # Initialize the multi-head encoder-decoder attention mechanism
        self.encoder_decoder_attention = MultiHeadAttention(d_model, heads)
        # Initialize the first add & norm (layer normalization) step
        self.norm1 = AddNorm(d_model)
        # Initialize the position-wise feed-forward network
        self.ffn = FeedForwardNetwork(d_model, dropout, forward_expansion)
        # Initialize the second add & norm (layer normalization) step
        self.norm2 = AddNorm(d_model)

    def forward(self, value, key, query, mask):
        # Pass the input through the multi-head self-attention mechanism
        self_attention = self.self_attention(value, key, query, mask)
        # Pass the self-attention output through the first add & norm (layer normalization) step
        x = self.norm1(query + self_attention)
        # Pass the output through the multi-head encoder-decoder attention mechanism
        encoder_decoder_attention = self.encoder_decoder_attention(value, key, x, mask)
        # Pass the encoder-decoder attention output through the second add & norm (layer normalization) step
        x = self.norm2(x + encoder_decoder_attention)
        # Pass the output through the position-wise feed-forward network
        ffn = self.ffn(x)
        # Pass the ffn output through the third add & norm (layer normalization) step
        output = self.norm3(x + ffn)

        return output
