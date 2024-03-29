import torch
from torch import nn


class AddNorm(nn.Module):
    """
    Add & Norm Layer in the paper "Attention is All You Need"

    It is used to add the residual connection and apply layer normalization
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): The dimensionality of the input embeddings.
        """
        super(AddNorm, self).__init__()

        # Define the layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            residual (Tensor): Residual tensor of shape (batch_size, seq_len, d_model)

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
