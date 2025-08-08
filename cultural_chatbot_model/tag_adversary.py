"""
Adversary Code

Uses Deepseek 7b embeddings as an input, and outputs a predicted 4-dim vector representing the cultural values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


class TagAdversary(nn.Module):
    """Takes a 4096-dim vector and squashes first hidden dimensions, then the output dimension."""

    def __init__(self, input_dim=4096, hidden_dim1=2048, hidden_dim2=364, output_dim=4):
        super().__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """Forward pass that converts text into a 4-dim vector representing cultural tags"""
        x = x.to(dtype=torch.float)
        x = x.to(next(self.parameters()).device)
        x = f.relu(self.hidden_layer1(x))
        x = f.relu(self.hidden_layer2(x))
        predicted_tag = f.sigmoid(self.output_layer(x))
        return predicted_tag

