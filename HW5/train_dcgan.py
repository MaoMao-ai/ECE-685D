import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# -----------------------------
# 1. Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 4, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).view(-1)

# -----------------------------
# 2. Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# -----------------------------
# 3. Training
# -----------------------------
def save_samples(G, device, epoch):
    G.eval()
    z = torch.randn(12, 100).to(device)
    fake = G(z).detach().cpu()

    fake = (fake + 1) / 2  # [-1,1] → [0,1]

    grid = torch.cat([img.squeeze(0) for img in fake], dim=1)
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.savefig(f"samples_epoch_{epoch}.png")
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    D = Discriminator().to(device)
    G = Generator().to(device)

    criterion = nn.BCELoss()
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 20

    for ep in range(epochs):
        print("Start training epoch", ep)
        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)

            # Train D
            D.zero_grad()
            labels_real = torch.full((bs,), 0.9).to(device) 
            labels_fake = torch.zeros(bs).to(device)

            out_real = D(real)
            loss_real = criterion(out_real, labels_real)

            z = torch.randn(bs, 100).to(device)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_fake = criterion(out_fake, labels_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # Train G
            G.zero_grad()
            out_fake2 = D(fake)
            loss_G = criterion(out_fake2, labels_real)
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {ep}: LossD={loss_D:.4f}, LossG={loss_G:.4f}")
        save_samples(G, device, ep)

    torch.save(G.state_dict(), "G_hw5.pth")
    torch.save(D.state_dict(), "D_hw5.pth")
    print("Training done.")


if __name__ == "__main__":
    main()