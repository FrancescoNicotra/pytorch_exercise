import torch
from torchvision.transforms import v2
from typing import Tuple, Sequence, Optional, Dict
import torch.nn as nn
from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

class ImageClassifierTransforms:
    def __init__(
        self,
        size: Sequence[int],
        antialias: bool,
        scale: Tuple[float, float],
        p: float = 0.5,
        normalize: bool = True,
        normalizeObj: Optional[Dict[str, Sequence[float]]] = None,
        toImage: bool = True,
        log_dir: Optional[str] = None
    ) -> None:
        transforms = [
            v2.RandomResizedCrop(size=size, antialias=antialias, scale=scale),
            v2.RandomHorizontalFlip(p=p),
        ]

        if toImage:
            transforms.append(v2.ToImage())

        transforms.append(v2.ToDtype(torch.float32, scale=True))

        if normalize:
            if normalizeObj is None:
                normalizeObj = {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            transforms.append(v2.Normalize(normalizeObj["mean"], normalizeObj["std"]))

        if log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.transform = v2.Compose(transforms)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)

    def create_model(self, num_class: int = 10) -> models.ResNet:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_class)
        return model

    def optimize_model(self, model: models.ResNet, learning_rate: float = 0.001) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
        return optimizer

    def get_dataloaders(self, batch_size: int = 32, root: str = "models/cifar10") -> Tuple[DataLoader, DataLoader]:
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=self.transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=self.transform)

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader

    def train_one_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int = 0) -> float:
        if self.__check_if_exists_a_trained_model("models/trained/model.pth"):
            model.load_state_dict(torch.load("models/trained/model.pth"))
            return 0.0
        model.train()
        total_loss: float = 0.0

        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (self.writer):
                self.writer.add_scalar("Loss/train_batch", loss.item(), step)

        self.__save_model(model, "models/trained/model.pth")
        if self.writer:
            self.writer.add_scalar("Loss/train_epoch", total_loss / len(dataloader), global_step=epoch)
        return total_loss / len(dataloader)

    def evaluate(self, model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int = 0) -> float:
        model.eval()
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        if self.writer:
            self.writer.add_scalar("Accuracy/test", correct / total, global_step=epoch)

        return correct / total

    def __save_model(self, model: nn.Module, path: str) -> None:
       os.makedirs(os.path.dirname(path), exist_ok=True)
       torch.save(model.state_dict(), path)

    def __check_if_exists_a_trained_model(self, path: str) -> bool:
        try:
            with open(path, 'r'):
                return True
        except FileNotFoundError:
            return False

    def load_model(self, path: str) -> nn.Module:
        model = self.create_model()
        if self.__check_if_exists_a_trained_model(path):
            model.load_state_dict(torch.load(path))
        return model

    def get_model(self, path: str) -> nn.Module:
        model = self.load_model(path)
        return model