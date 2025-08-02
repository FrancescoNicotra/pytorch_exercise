# utils function for LSTM model
import torch
from typing import Tuple
from torch import Tensor
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import os


def normalize_tensor(tensor: Tensor) -> Tensor:
    """
    Normalize a tensor to have zero mean and unit variance.

    Args:
            tensor (Tensor): Input tensor to normalize.

    Returns:
            Tensor: Normalized tensor.
    """
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std


def get_device() -> str:
    """
    Get the device to be used for PyTorch operations.

    Returns:
            str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data_for_model(data: pd.DataFrame, sequence_length: int) -> Tuple[Tensor, Tensor]:
    """
    Prepare data for the LSTM model by converting it to tensors and creating sequences.

    Args:
            data (pd.DataFrame): Input data as a pandas DataFrame.
            sequence_length (int): Length of the sequences to create.

    Returns:
            Tuple[Tensor, Tensor]: Tuple containing input tensor and target tensor.
    """
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    x = data_tensor[:-sequence_length]
    y = data_tensor[sequence_length:]
    return x, y


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> None:
    """
    Train the LSTM model.

    Args:
            model (nn.Module): The LSTM model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (str): Device to run the training on ('cpu' or 'cuda').

    Returns:
            None
    """
    model.train()
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: str) -> float:
    """
    Evaluate the LSTM model on the test data.

    Args:
            model (nn.Module): The LSTM model to evaluate.
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            criterion (nn.Module): Loss function.
            device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
            float: Average loss over the test dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def save_model(model: nn.Module, file_path: str) -> None:
    """
    Save the trained LSTM model to a file.

    Args:
            model (nn.Module): The LSTM model to save.
            file_path (str): Path to save the model file.

    Returns:
            None
    """
    torch.save(model.state_dict(), file_path)


def load_model(model: nn.Module, file_path: str) -> None:
    """
    Load a trained LSTM model from a file.

    Args:
            model (nn.Module): The LSTM model to load the state into.
            file_path (str): Path to the model file.

    Returns:
            None
    """
    model.load_state_dict(torch.load(file_path, map_location=get_device()))
    model.eval()


def check_if_model_exists(file_path: str) -> bool:
    """
    Check if a model file exists.

    Args:
            file_path (str): Path to the model file.

    Returns:
            bool: True if the model file exists, False otherwise.
    """
    return os.path.exists(file_path)


def plot_predictions(predictions: torch.Tensor, actual: torch.Tensor, title: str = "Predictions vs Actual") -> None:
    """
    Plot predictions against actual values.

    Args:
                    predictions (Tensor): Predicted values from the model.
                    actual (Tensor): Actual values to compare against.
                    title (str): Title of the plot.

    Returns:
                    None
    """
    # check if has CUDA support
    if predictions.is_cuda:
        predictions = predictions.cpu()
    if actual.is_cuda:
        actual = actual.cpu()
    plt.figure(figsize=(10, 5))
    plt.plot(actual.cpu().numpy(), label='Actual', color='blue')
    plt.plot(predictions.cpu().numpy(), label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def save_plot(predictions: Tensor, actual: Tensor, file_path: str) -> None:
    """
    Save a plot of predictions against actual values to a file.

    Args:
                    predictions (Tensor): Predicted values from the model.
                    actual (Tensor): Actual values to compare against.
                    file_path (str): Path to save the plot image.

    Returns:
                    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual.cpu().numpy(), label='Actual', color='blue')
    plt.plot(predictions.cpu().numpy(), label='Predicted', color='red')
    plt.title("Predictions vs Actual")
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(file_path)
    plt.close()
