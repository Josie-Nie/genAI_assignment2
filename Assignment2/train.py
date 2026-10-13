"""Train the specified CNN on CIFAR-10, saving weights for inference."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from app.model import AssignmentCNN
from app.preprocess import image_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-path", default="models/cifar10_cnn.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    train_ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=image_transform)
    test_ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=image_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = -1.0
    checkpoint = Path(args.model_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predicted = model(images).argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{args.epochs}: train loss={running_loss / len(train_ds):.4f} test accuracy={accuracy:.2%}")
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.cpu().state_dict(), checkpoint)
            model.to(device)
            print(f"Saved best model at {checkpoint}")
    print(f"Best test accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
