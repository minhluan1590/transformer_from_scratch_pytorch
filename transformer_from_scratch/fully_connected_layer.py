import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """
    Freed Forward Network in the paper "Attention is All You Need"
    """

    def __init__(self, d_model, dropout: float or None = None, forward_expansion=4):
        """
        Args:
            d_model (int): The dimensionality of the input embeddings.
            dropout (float or None): The dropout probability of the feed-forward network. Defaults to None.
            forward_expansion (int): The forward expansion factor of the feed-forward network. Defaults to 4.
        """
        super(FeedForwardNetwork, self).__init__()

        # Define the first Linear layer
        self.fc_1 = nn.Linear(d_model, d_model * forward_expansion)

        # Define the relu activation
        self.relu = nn.ReLU()

        # Define the second Linear layer
        self.fc_2 = nn.Linear(d_model * forward_expansion, d_model)

        # Define the dropout layer if dropout is not None
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """

        # Apply the first Linear layer
        # The shape of output is (batch_size, seq_len, d_ff)
        output = self.fc_1(x)

        # Apply the relu activation
        # The shape of output is (batch_size, seq_len, d_ff)
        output = self.relu(output)

        # Apply the dropout layer if dropout is not None
        if self.dropout is not None:
            # The shape of output is (batch_size, seq_len, d_ff)
            output = self.dropout(output)

        # Apply the second Linear layer
        # The shape of output is (batch_size, seq_len, d_model)
        output = self.fc_2(output)

        return output
