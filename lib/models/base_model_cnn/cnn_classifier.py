from flatten import Flatten
from separator_conv_1d import SeparatorConv1d
from torch import nn


class CnnClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size,  linear_channels, dropout=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            SeparatorConv1d(input_channels, 32, kernel_size, 2, 3, dropout=dropout),
            SeparatorConv1d(32, 64, kernel_size, 4, 2, dropout=dropout),
            SeparatorConv1d(64, 128, kernel_size, 4, 2, dropout=dropout),
            SeparatorConv1d(128, 256, kernel_size, 4, 2),

            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(linear_channels, 64),

            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, input):
        return self.layers(input)
