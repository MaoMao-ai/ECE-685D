import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

###########################################
# Device selection
###########################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

###########################################
# Weight initialization (Required for DCGAN)
###########################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

###########################################
# Discriminator (Stable Version)
###########################################
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 1×28×28 → 64×14×14
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64×14×14 → 128×7×7
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128×7×7 → 256×4×4
            nn.Conv2d(128, 256, 4, 1, 0), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256×4×4 → 1×1×1
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

###########################################
# Generator (Stable Version)
###########################################
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(

            # 1. Fully connected → 7×7×256
            nn.Linear(z_dim, 7 * 7 * 256),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 7, 7)),

            # 2. 7×7×256 → 14×14×128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 3. 14×14×128 → 28×28×64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4. 28×28×64 → 28×28×32
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 5. 28×28×32 → 28×28×1
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


###########################################
# Save generated samples
###########################################
def save_samples(G, epoch):
    G.eval()
    z = torch.randn(12, 100).to(device)
    fake = G(z).detach().cpu()

    fake = (fake + 1) / 2  # [-1,1] → [0,1]
    grid = torch.cat([img.squeeze(0) for img in fake], dim=1)

    os.makedirs("samples", exist_ok=True)
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.savefig(f"samples/epoch_{epoch}.png")
    plt.close()

###########################################
# Training
###########################################
def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.CenterCrop(28),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    D = Discriminator().to(device)
    G = Generator().to(device)

    # DCGAN REQUIREMENT: APPLY WEIGHT INIT
    D.apply(weights_init)
    G.apply(weights_init)

    criterion = nn.BCELoss()
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 20

    for ep in range(epochs):
        print("Start training epoch", ep)

        for real, _ in loader:
            ###########################################
            # Check real shape once
            ###########################################
            assert real.shape[-1] == 28 and real.shape[-2] == 28, \
                f"ERROR: real image shape {real.shape}, MUST be 1×28×28"

            real = real.to(device)
            bs = real.size(0)

            ###########################################
            # Train D
            ###########################################
            D.zero_grad()
            labels_real = torch.ones(bs).to(device)
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

            ###########################################
            # Train G
            ###########################################
            G.zero_grad()
            out_fake2 = D(fake)
            loss_G = criterion(out_fake2, labels_real)
            loss_G.backward()
            opt_G.step()

        ###########################################
        # Epoch summary
        ###########################################
        print(f"Epoch {ep}: LossD={loss_D:.4f}, LossG={loss_G:.4f}")
        save_samples(G, ep)

    torch.save(G.state_dict(), "G_hw5.pth")
    torch.save(D.state_dict(), "D_hw5.pth")
    print("Training done.")\
    
if __name__ == "__main__":
    main()