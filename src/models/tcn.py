import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()           
        self.chomp_size = chomp_size

    def forward(self, x):
        # recorta self.chomp_size valores del final en la dimensión temporal
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        y = self.relu1(self.chomp1(self.conv1(x)))
        y = self.relu2(self.chomp2(self.conv2(y)))
        res = x if self.downsample is None else self.downsample(x)
        return y + res

class TCNForecaster(nn.Module):
    def __init__(self, n_feats, num_channels=[64,64,64], kernel_size=3, horizon=100):
        super().__init__()
        layers = []
        in_ch = n_feats
        for i, out_ch in enumerate(num_channels):
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                     dilation=2**i, padding=(kernel_size-1)*(2**i))]
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], horizon)

    def forward(self, x):
        # x: B×T×F → B×F×T
        t = x.transpose(1,2)
        y = self.tcn(t)            # B×C×T
        last = y[:, :, -1]         # B×C
        return self.linear(last)   # B×horizon