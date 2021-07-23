import torch
import torch.nn as nn

class EpochEncoderCnn(nn.Module):

    def __init__(self, input_shape):

        self.input_shape = input_shape

        self.use_cuda = torch.cuda.is_available()

        self.features = nn.Sequential(

        )
