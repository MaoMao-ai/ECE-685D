import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

#######################################
# Load pretrained Discriminator
#######################################
from train_dcgan import Discriminator

# Load discriminator (same architecture as training)
D = Discriminator().to(device)
D.load_state_dict(torch.load("D_hw5.pth", map_location=device))
D.eval()

#######################################
# Remove last conv (feature extractor)
#######################################
# Original net:
# [Conv1, LeakyReLU,
#  Conv2, BN, LeakyReLU,
#  Conv3, BN, LeakyReLU,
#  Conv4(256->1), Sigmoid]

# We remove last two layers: Conv4 + Sigmoid
D_features = nn.Sequential(*list(D.net.children())[:-2]).to(device)
D_features.eval()

# Confirm output dimension
test_input = torch.randn(1, 1, 28, 28).to(device)
feat = D_features(test_input)
print("Feature map shape:", feat.shape)  # expected: (1, 256, 4, 4)


#######################################
# Linear classifier (4096 → 10)
#######################################
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        return self.fc(x)

clf = LinearClassifier().to(device)


#######################################
# MNIST dataset
#######################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
    transforms.CenterCrop(28),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

#######################################
# Use only 10% of training set
#######################################
train_size = int(0.1 * len(train_dataset))
train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
train_subset = Subset(train_dataset, train_indices)

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

#######################################
# Train classifier
#######################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(), lr=0.001)

epochs = 10

print("Start training classifier...")

for ep in range(epochs):
    clf.train()
    correct, total, train_loss = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            feat = D_features(x)

        optimizer.zero_grad()
        out = clf(feat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    acc = correct / total * 100
    print(f"Epoch {ep}: Train Loss={train_loss:.4f}, Train Acc={acc:.2f}%")

#######################################
# Test classifier
#######################################
clf.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        feat = D_features(x)
        out = clf(feat)

        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

test_acc = correct / total * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
