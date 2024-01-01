import torch
from torch import nn


class AddNorm(nn.Module):
    """
    Add & Norm Layer in the paper "Attention is All You Need"

    It is used to add the residual connection and apply layer normalization
    """

    def __init__(self, d_model):
        """
        Args:
            d_model: Embedding size of the input
        """
        super(AddNorm, self).__init__()

        # Define the layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            residual: Residual tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Add the input tensor and residual tensor
        # The shape of output is (batch_size, seq_len, d_model)
        output = x + residual

        # Apply layer normalization
        # The shape of output is (batch_size, seq_len, d_model)
        output = self.layer_norm(output)

        return output
