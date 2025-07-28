import torch
from torchvision.transforms import v2

#define class to create torch transforms for image classification
class ImageClassifierTransforms:
    def __init__(self) -> None:
        self.transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)