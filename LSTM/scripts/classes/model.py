# LSTMModel for PyTorch
# This file contains the LSTM model definition for sequence prediction tasks.
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size),
                              representing a batch of sequences.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), containing the predicted value
                          for each sequence in the batch (e.g. the value for the next time step).
        """
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def train(self, mode=True):
        return super().train(mode)

    def save(self, path):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def eval(self):
        """
        Set the model to evaluation mode.
        This is used during inference to disable dropout and batch normalization.
        """
        return super().eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Predicted output tensor.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def cpu(self):
        return super().cpu()

    def cuda(self):
        return super().cuda()
