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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoding_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        x = self.encoding_layers(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoding_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.decoding_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder, latent_dim=8):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(.5*logvar))
        t = self.Decoder(z)
        return t, mean, logvar

class DetectorDataset(Dataset):
    def __init__(self, counts):
        self.data = counts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def loss_function(args, output, target):
    x, mu, logvar = output

    mse_loss = nn.functional.mse_loss(x, target)
    kld_loss = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + args.beta * kld_loss

def train(args, model, loader, optimizer, epoch, no_print=False):
    initial_time = time()
    model.train()
    train_losses = []
    for batch_index, counts in enumerate(loader):
        optimizer.zero_grad()
        output = model(counts)
        loss = loss_function(args, output, counts)
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
            losses.append(loss_function(args, output, counts).item())
            if not no_print:
                print(f"Testing progress [{batch_index*len(counts)}/{len(loader.dataset)}] {100.*batch_index*len(counts)/len(loader.dataset):.2f}%", end="\r")
    test_loss = np.mean(losses)
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Testing\t\t\t\tAverage loss: {test_loss:.6f}\t({time()-initial_time:.2f}s elapsed)")
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

def generate_ax(args, model, ax, spots, minor=False):
    model.eval()
    with torch.no_grad():
        counts = model.Decoder(torch.randn(model.latent_dim))
    
    ax.plot(spots, counts, color = "tab:orange")

    ax.set_xlim(left=1, right=33*4)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=33))
    if minor:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

def generate_graphs(args, model, number=1):

    fig, axs = plt.subplots(number, number, figsize=(1.5*number,1.5*number), sharex="all", layout="compressed")

    spots = np.arange(1, 1+33*4)

    if number == 1:
        generate_ax(args, model, axs, spots, minor=True)
    else:
        for ax in axs.ravel():
            generate_ax(args, model, ax, spots, minor=False)

    fig.supxlabel("Detector position")
    fig.supylabel("Photon count")

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
    parser.add_argument("--save", type=bool, default=True,
                        help="input whether to save model (default=True)")
    parser.add_argument("--load", type=bool, default=False,
                        help="enable load mode, loading from save-name (default=False)")
    parser.add_argument("--print-number", type=int, default=1,
                        help="input printed outputs number, which will be squared (default=1)")
    parser.add_argument("--latent-dim", type=int, default=8,
                        help="input latent space dimensionality (default=8)")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="input hidden layers dimensionality (default=32)")
    parser.add_argument("--beta", type=float, default=.5,
                        help="input kldiv loss weight (default=0.5)")
    
    return parser

def main():
    main_time = time()
    print("Program start")
    
    parser = create_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    enc = Encoder(input_dim=132, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    dec = Decoder(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, output_dim=132)
    model = VAE(Encoder=enc, Decoder=dec, latent_dim=args.latent_dim)
    if not args.load:
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
        print("Data loaded")
        
        match args.optim:
            case "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            case "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        train_losses, test_losses = [], []
        print("Training start")
        for epoch in range(1, args.epochs+1):
            train_log = train(args, model, train_loader, optimizer, epoch)
            test_loss = test(args, model, test_loader)

            train_losses.append(train_log)
            test_losses.append(test_loss)
        print("Training end")

        if args.save:
            torch.save(model.state_dict(), f"vae/{args.folder_name}_models/{args.save_name}_model.pt")
            print("Model saved")
        
        paint_losses(args, np.ravel(train_losses), test_losses, train_loader, test_loader)
        print("Losses painted")

    else:
        model.load_state_dict(torch.load(f"vae/{args.folder_name}_models/{args.save_name}_model.pt"))
        model.eval()

    generate_graphs(args, model, number=args.print_number)
    print("Graphs generated")

    main_total = time()-main_time
    main_days = int(main_total // 86400)
    main_hours = int((main_total % 86400) // 3600)
    main_minutes = int(((main_total % 86400) % 3600) // 60)
    main_seconds = int(((main_total % 86400) % 3600) % 60)
    print(f"Program complete ({main_days:02}:{main_hours:02}:{main_minutes:02}:{main_seconds:02} elapsed)")

if __name__ == "__main__":
    main()
