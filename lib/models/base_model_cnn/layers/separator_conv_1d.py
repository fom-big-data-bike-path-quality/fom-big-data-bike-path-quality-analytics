from torch import nn


class _SeparatorConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SeparatorConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dropout=None,
                 activation_layer=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert dropout is None or (0.0 < dropout < 1.0)
        layers = [_SeparatorConv1d(in_channels, out_channels, kernel_size, stride, padding)]
        if activation_layer:
            layers.append(activation_layer())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
