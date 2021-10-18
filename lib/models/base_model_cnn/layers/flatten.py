from torch import nn


class Flatten(nn.Module):

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, input):
        if self.keep_batch_dim:
            return input.view(input.size(0), -1)
        return input.view(-1)
