"""
Neural network that converts BERT Style Embeddings from MiniLLM into a 4-dim vector that represents
a four cultural tags based on Hofstede's Cultural Dimensions and Self-Construal Theory.

The 4-dim vector [a, b, c, d] means:
    --> a = power distance
    --> b = individualism
    --> c = independent self-construal
    --> d = uncertainty avoidance

NOTE: THIS IS A COPY FROM THE CULTURAL CHATBOT MODEL FOLDER. USED STRICTLY FOR TAGGER BACKEND.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


class Tagger(nn.Module):
    """defines the tagger class which takes a 384-dim BERT style embedding vector and outputs a 4-dim vector"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=4):
        super().__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """forward pass of the neural network. returns list form of vector.

        x must be some kind of 384-dim BERT style embedding vector.
        """
        with torch.no_grad():
            x = x.clone().detach()  # Ensures it's not an inference-only tensor

            x = f.relu(self.hidden_layer1(x))
            cultural_vector = torch.sigmoid(self.hidden_layer2(x))

            return cultural_vector.squeeze(0).tolist()
