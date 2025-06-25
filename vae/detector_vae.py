import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms, datasets #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt
from utils.early_stopper import EarlyStopper
from time import time
import numpy as np
import pandas as pd # type: ignore
import argparse
from scipy.signal.windows import gaussian

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(132, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU()
        )
        self.mean_layer = nn.Linear(16, 2)
        self.logvar_layer = nn.Linear(16, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 132),
            nn.ReLU()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        t = self.decode(z)
        return t

class DetectorDataset(Dataset):
    def __init__(self, counts):
        self.data = counts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def loss_function(x, target):
    gw = torch.from_numpy(gaussian(33,1))
    ref_gaussian = torch.cat((gw,gw,gw,gw))
    
    mse_loss = nn.functional.mse_loss(x, target)
    kld_loss = nn.functional.kl_div(x, ref_gaussian, reduction="batchmean")

    return mse_loss + kld_loss

def train(args, model, loader, optimizer, epoch, no_print=False):
    initial_time = time()
    model.train()
    train_losses = []
    for batch_index, counts in enumerate(loader):
        optimizer.zero_grad()
        output = model(counts)
        loss = loss_function(output, counts)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if not no_print and batch_index % args.logging_interval == 0:
            print(f"Training epoch {epoch} [{batch_index*len(counts)}/{len(loader.dataset)}]\tLoss: {loss.item():.6f}", end="\r")
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Training epoch {epoch} complete\tAverage loss: {np.mean(train_losses):.6f}\t({time()-initial_time:.2f}s elapsed)")
    return train_losses

def test(args, model, loader, no_print=False):
    initial_time = time()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, counts in enumerate(loader):
            output = model(counts)
            losses.append(loss_function(output, counts).item())
            pred = output.argmax(dim=1, keepdim=True)
            if not no_print:
                print(f"Testing progress [{batch_index*len(counts)}/{len(loader.dataset)}] {100.*batch_index*len(counts)/len(loader.dataset):.2f}%", end="\r")
    test_loss = np.mean(losses)
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Testing\t\tAverage loss: {test_loss:.6f}\t({time()-initial_time:.2f}s elapsed)")
    return test_loss

def paint_losses(args, train_losses, test_losses, train_loader, test_loader):
    fig, ax = plt.subplots()

    train_spots = np.arange(1, 1+args.epochs*(1+(len(train_loader.dataset)//args.train_batch_size))) * args.train_batch_size
    test_spots = np.arange(1, 1+len(test_losses)) * len(train_loader.dataset)

    ax.plot(train_spots, train_losses, ls="solid", color="tab:blue", label="Train loss")
    ax.plot(test_spots, test_losses, ls="None", marker="o", color="tab:orange", label="Test loss")

    ax.xaxis.set_major_locator(plt.MultipleLocator(len(train_loader.dataset)))
    
    ax.set_xlim(left=0)

    ax.legend(loc="best")
    
    fig.savefig(f"vae/{args.folder_name}_figs/{args.save_name}_losses.pdf", dpi=300, bbox_inches="tight")

def generate_graph(args, model, mean, var):
    fig, ax = plt.subplots()

    spots = np.arange(1, 1+33*4)

    model.eval()
    with torch.no_grad():
        counts = model.decode(torch.tensor((mean, var)))
    
    ax.plot(spots, counts)

    ax.set_xlabel("Detector position")
    ax.set_ylabel("Photon count")

    ax.set_xlim(left=1, right=33*4)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=33))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    fig.savefig(f"vae/{args.folder_name}_gen/{args.save_name}_gen.pdf", dpi=300, bbox_inches="tight")

def create_parser():
    parser = argparse.ArgumentParser(description="Variational Autoencoder")
    
    # File settings
    parser.add_argument("--folder-name", type=str, default="vae",
                        help="input folder for storing figures (default=vae)")
    parser.add_argument("--save-name", type=str, default="vae",
                        help="input file save name (default=vae)")

    # Training settings
    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="input training batch size (default=64)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="input testing batch size (default=1000)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="input number of epochs (default=100)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="input learning rate (default=0.01)")
    parser.add_argument("--random-seed", type=int, default=1,
                        help="input random seed (default=1)")
    parser.add_argument("--logging-interval", type=int, default=50,
                        help="input logging interval (default=50)")
    parser.add_argument("--optim", type=str, default="Adam",
                        help="input optimizer (default=Adam)")
    parser.add_argument("--seed", type=int, default=1,
                        help="input random seed (default=1)")
    parser.add_argument("--save", type=bool, default=False,
                        help="input whether to save model (default=False)")
    
    return parser

def main():
    main_time = time()
    
    parser = create_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    df = pd.DataFrame(pd.read_pickle("vae/simulated_events.pickle"))
    photon_counts = torch.tensor(list(df[df.columns[-1]].values))
    
    train_counts, test_counts = train_test_split(photon_counts, test_size=.2, random_state=args.seed)

    train_loader = DataLoader(
        DetectorDataset(train_counts),
        batch_size=args.train_batch_size, shuffle=True
    )

    test_loader = DataLoader(
        DetectorDataset(test_counts),
        batch_size=args.test_batch_size, shuffle=True
    )

    model = VAE()
    match args.optim:
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_losses, test_losses = [], []

    for epoch in range(1, args.epochs+1):
        train_log = train(args, model, train_loader, optimizer, epoch)
        test_loss = test(args, model, test_loader)

        train_losses.append(train_log)
        test_losses.append(test_loss)

    if args.save:
        torch.save(model.state_dict(), f"vae/{args.folder_name}_models/{args.save_name}_model.pt")

    paint_losses(args, np.ravel(train_losses), test_losses, train_loader, test_loader)

    mean, var = 0., 1.
    generate_graph(args, model, mean, var)

    main_total = time()-main_time
    main_days = int(main_total // 86400)
    main_hours = int((main_total % 86400) // 3600)
    main_minutes = int(((main_total % 86400) % 3600) // 60)
    main_seconds = int(((main_total % 86400) % 3600) % 60)
    print(f"Program complete ({main_days:02}:{main_hours:02}:{main_minutes:02}:{main_seconds:02} elapsed)")

if __name__ == "__main__":
    main()
