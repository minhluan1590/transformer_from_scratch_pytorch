import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Self Attention Layer in the paper "Attention is All You Need"
    """

    def __init__(self, d_model: int, heads: int):
        """
        Args:
            d_model (int): The dimensionality of the input embeddings.
            heads (int): The number of attention heads.
        """

        super(MultiHeadAttention, self).__init__()

        # Store the embed size and heads
        self.d_model = d_model
        self.heads = heads

        # Calculate the head dimension
        self.head_dim = d_model // heads

        assert (
            self.head_dim * heads == d_model
        ), "Embedding size needs to be divisible by heads"

        # Define the query, key and value matrices for all heads
        # Bias set to False to avoid adding bias in the Linear layer.
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final Linear layer to output the attention output
        self.fc_out = nn.Linear(
            heads * self.head_dim, d_model
        )  # or we can write nn.Linear(d_model, d_model) because heads * self.head_dim = d_model

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            value (Tensor): Value matrix of shape (batch_size, seq_len, d_model)
            key (Tensor): Key matrix of shape (batch_size, seq_len, d_model)
            query (Tensor): Query matrix of shape (batch_size, seq_len, d_model)
            mask (Tensor): Mask to apply to the attention layer. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """

        # Get the batch size
        batch_size = query.shape[0]

        # Get the sequence length
        seq_len = query.shape[1]

        # Split the embedding into self.heads pieces
        value = value.reshape(batch_size, seq_len, self.heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.heads, self.head_dim)
        query = query.reshape(batch_size, seq_len, self.heads, self.head_dim)

        # Calculate the query, key and value matrices for all heads in a single batch
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Calculate the energy without using einsum
        # The query shape is (batch_size, seq_len, heads, head_dim)
        # The key shape is (batch_size, seq_len, heads, head_dim)
        # We calculate the dot product of query and key for each batch and head
        # The shape of energy will be (batch_size, heads, seq_len, seq_len)
        energy = torch.matmul(query, key.transpose(2, 3))

        # Calculate scaled energy
        energy = energy / (self.d_model ** (1 / 2))

        # Apply the mask if it is not None
        # The shape of energy is (batch_size, heads, seq_len, seq_len)
        # The shape of mask is (batch_size, 1, seq_len, seq_len)
        # We add the mask with energy and apply softmax to the result
        # The shape of attention is (batch_size, heads, seq_len, seq_len)
        if mask is not None:
            attention = torch.softmax(energy + mask, dim=-1)
        else:
            attention = torch.softmax(energy, dim=-1)

        # Calculate the output of the attention layer
        # The shape of value is (batch_size, seq_len, heads, head_dim)
        # The shape of attention is (batch_size, heads, seq_len, seq_len)
        # We multiply value and attention and transpose the result
        # The shape of attention_output is (batch_size, seq_len, heads, head_dim)
        attention_output = torch.matmul(attention, value)

        # Reshape the attention output
        # The shape of attention_output is (batch_size, seq_len, heads, head_dim)
        # We reshape it to (batch_size, seq_len, heads * head_dim)
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.heads * self.head_dim
        )

        return self.fc_out(attention_output)
