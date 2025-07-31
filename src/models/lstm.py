import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_feats, hidden_size=64, num_layers=2, horizon=100):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: B×T×F
        out, _ = self.lstm(x)           # out: B×T×H
        last = out[:, -1, :]            # B×H
        y_hat = self.linear(last)       # B×horizon
        return y_hat