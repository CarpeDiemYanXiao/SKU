import torch
import torch.nn as nn

class flatten_feature_extractor(nn.Module):
    def __init__(self, state_dim, hidden_dim = 64):
        super(flatten_feature_extractor, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)

    def forward(self,x):
        y = self.fc(x)
        return y 
