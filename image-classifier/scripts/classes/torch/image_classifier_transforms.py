import torch
from torchvision.transforms import v2
from typing import Tuple, Sequence
from typing import Optional, Dict, Sequence

class ImageClassifierTransforms:
    def __init__(
        self,
        size: Sequence[int],
        antialias: bool,
        scale: Tuple[float, float],
        p: float = 0.5,
        normalize: bool = True,
        normalizeObj: Optional[Dict[str, Sequence[float]]] = None,
        toImage: bool = True
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
            transforms.append(v2.Normalize(**normalizeObj))

        self.transform = v2.Compose(transforms)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)
