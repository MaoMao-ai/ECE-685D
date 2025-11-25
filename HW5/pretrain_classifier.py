import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from dcgan_hw5 import Discriminator


# Remove final layer (probability output)
class DiscriminatorFeature(nn.Module):
    def __init__(self):
        super().__init__()

        D_full = Discriminator()
        D_full.load_state_dict(torch.load("D_hw5.pth", map_location="cpu"))

        # remove last conv and sigmoid
        layers = list(D_full.model.children())[:-2]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim=512 * 4 * 4, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # 10% subset
    idx = np.random.choice(len(train), len(train) // 10, replace=False)
    train_subset = Subset(train, idx)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    feature_extractor = DiscriminatorFeature().to(device)
    feature_extractor.eval()

    classifier = LinearClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train classifier
    for ep in range(5):
        classifier.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = feature_extractor(x)

            pred = classifier(feat)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        print(f"Epoch {ep}: Train Acc = {correct/total:.4f}")

    # Evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            feat = feature_extractor(x)
            pred = classifier(feat)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    print("Test Acc:", correct / total)


if __name__ == "__main__":
    main()