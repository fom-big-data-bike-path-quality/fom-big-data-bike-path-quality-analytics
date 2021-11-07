from torch import nn
from torch import zeros


class LstmClassifier(nn.Module):
    def __init__(self, device, input_size, hidden_dimension, layer_dimension, num_classes, dropout=0.5):
        super().__init__()

        self.device = device

        self.hidden_dimension = hidden_dimension
        self.layer_dimension = layer_dimension

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dimension,
            num_layers=layer_dimension,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dimension, num_classes)

    def forward(self, input):
        h0 = zeros(self.layer_dimension, input.size(0), self.hidden_dimension).requires_grad_().to(self.device)
        c0 = zeros(self.layer_dimension, input.size(0), self.hidden_dimension).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.linear(out)

        return out
