import torch
from scripts.classes.torch.image_classifier_transforms import ImageClassifierTransforms
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    plt.rcParams["savefig.bbox"] = "tight"
    transforms = ImageClassifierTransforms()
    datasets = CIFAR10(root="./models/cifar10", download=True, train=True, transform=transforms)
    random_index = torch.randint(0, len(datasets), (1,)).item()
    img, label = datasets[int(random_index)]
    print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
    transformed_img = transforms(img)
    plt.imshow(transformed_img.permute(1, 2, 0).detach().numpy())
    plt.axis("off")
    plt.show()