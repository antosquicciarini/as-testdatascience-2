import torch.nn as nn

class TransfForecaster(nn.Module):
    def __init__(self, n_feats, d_model=64, nhead=4, nlayers=3, horizon=100):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.linear = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: B×T×F
        x = self.input_proj(x)            # B×T×d_model
        enc = self.encoder(x)             # B×T×d_model
        last = enc[:, -1, :]              # B×d_model
        return self.linear(last)          # B×horizon