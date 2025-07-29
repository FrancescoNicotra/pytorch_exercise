import torch
from scripts.classes.torch.image_classifier_transforms import ImageClassifierTransforms
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs: int = 10
    classes = [
        "airplane",  # 0
        "automobile",  # 1
        "bird",  # 2
        "cat",  # 3
        "deer",  # 4
        "dog",  # 5
        "frog",  # 6
        "horse",  # 7
        "ship",  # 8
        "truck"  # 9
    ]


    transforms = ImageClassifierTransforms(
        size=[224, 224],
        antialias=True,
        scale=(0.08, 1.0),
        p=0.5,
        normalize=True,
        normalizeObj={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        toImage=True,
        log_dir="./logs"
    )

    model_path = "models/trained/model.pth"

    if transforms.load_model(model_path):
        print("Loaded pre-trained model.")
        image_path: str = "image-classifier/images/what-do-you-know-about-cats-1781120238.jpg"
        image = Image.open(image_path)
        tensor_img = transforms.transform(image).unsqueeze(0).to(device)
        model: torch.nn.Module = transforms.get_model(model_path)
        model.eval()
        with torch.no_grad():
            output = model(tensor_img)
            _, predicted = torch.max(output, 1)
            print(f"Predicted class: {predicted.item()} - {classes[int(predicted.item())]}")
    else:
        print("No pre-trained model found, starting training from scratch.")
        train_loader, test_loader = transforms.get_dataloaders(32, root="./models/cifar10")
        model = transforms.create_model(num_class=10).to(device)
        optimizer = transforms.optimize_model(model, learning_rate=1e-3)

        for epoch in range(num_epochs):
            loss = transforms.train_one_epoch(model, train_loader, optimizer, device, epoch)
            acc = transforms.evaluate(model, test_loader, device, epoch)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    if transforms.writer:
        transforms.writer.close()
