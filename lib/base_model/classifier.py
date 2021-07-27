import os
import sys

from torch import nn

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/base_model'),
    os.path.join(os.getcwd(), 'lib/base_model/layers'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

from flatten import Flatten
from separator_conv_1d import SeparatorConv1d


class Classifier(nn.Module):
    def __init__(self, input_channels, num_classes, dropout=.5):
        super().__init__()

        self.layers = nn.Sequential(
            SeparatorConv1d(input_channels, 32, 8, 2, 3, dropout=dropout),
            SeparatorConv1d(32, 64, 8, 4, 2, dropout=dropout),
            SeparatorConv1d(64, 128, 8, 4, 2, dropout=dropout),
            SeparatorConv1d(128, 256, 8, 4, 2),

            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(768, 64),

            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, input):
        return self.layers(input)