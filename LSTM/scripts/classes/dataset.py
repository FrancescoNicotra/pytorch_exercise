import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple


class Dataset(TorchDataset):
    def __init__(self, data_path: str, sequence_length: int):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.data = self.load_data()

    def load_data(self) -> Tensor:
        df = pd.read_csv(self.data_path)

        # Prende solo una colonna numerica (es. 'Close')
        # Puoi parametrizzarla se vuoi
        close_prices = df["Close"].values.astype("float32")
        return torch.tensor(close_prices).unsqueeze(1)  # shape (N, 1)

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # shape (seq_len, 1)
        x = self.data[index: index + self.sequence_length]
        # shape (1)
        y = self.data[index + self.sequence_length]
        return x, y
