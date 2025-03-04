
import torch
from torch import nn



#####################
# Bottlenecks
#####################


class BaseBottleneck(nn.Module):
    """Basic fcn bottleneck

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):
        """
        b = batch size
        h = hidden dimension
        z = latent dim
        
        input_dim: b x s x h
        output_dim: b x z
        """

        mu = self.fc1(h)
        logvar = self.fc2(h)

        return mu, logvar


