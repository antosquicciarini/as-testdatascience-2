import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, X, y, input_len, horizon, stride=72):
        self.X = X
        self.y = y
        self.input_len = input_len
        self.horizon = horizon
        self.stride = stride

    def __len__(self):
        return (len(self.y) - self.input_len - self.horizon) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x_window = self.X[
            start: start + self.input_len
            ]
        y_window = self.y[
            start + self.input_len: start + self.input_len + self.horizon
            ]
        return torch.tensor(x_window, dtype=torch.float32), \
            torch.tensor(y_window, dtype=torch.float32)
