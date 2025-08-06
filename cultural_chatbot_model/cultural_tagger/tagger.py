"""
Neural network that converts BERT Style Embeddings from MiniLLM into a 4-dim vector that represents
a four cultural tags based on Hofstede's Cultural Dimensions and Self-Construal Theory.

The 4-dim vector [a, b, c, d] means:
    --> a = power distance
    --> b = individualism
    --> c = independent self-construal
    --> d = uncertainty avoidance
"""
import torch.nn as nn
import torch.nn.functional as f


class Tagger(nn.Module):
    """defines the tagger class which takes a 384-dim BERT style embedding vector and outputs a 4-dim vector"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=4):
        super().__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """forward pass of the neural network. returns tuple where index 0 is the list-form of the vector and
        index 1 is the string-form used to attach to prompts.

        x must be some kind of sentence/word string.
        """
        x = f.relu(self.hidden_layer1(x))
        cultural_vector = f.sigmoid(self.hidden_layer2(x))\

        if len(cultural_vector.shape) == 1 or cultural_vector.shape[0] == 1:
            tag = cultural_vector.squeeze(0).tolist()
            tag_str = f"[pd={tag[0]:.2f}][indiv={tag[1]:.2f}][independent={tag[2]:.2f}][ua={tag[3]:.2f}]"
            return cultural_vector, tag_str  # single string

        else:
            cultural_tags = []
            for vec in cultural_vector:
                tag = vec.tolist()
                tag_str = f"[pd={tag[0]:.2f}][indiv={tag[1]:.2f}][independent={tag[2]:.2f}][ua={tag[3]:.2f}]"
                cultural_tags.append(tag_str)
            return cultural_vector, cultural_tags
