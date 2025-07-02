import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms, datasets #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd # type: ignore
import argparse
from pathlib import Path
import pickle

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoding_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, l):
        total_input = torch.cat((x,l), dim=1)
        x = self.encoding_layers(total_input)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoding_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z, l, dim=1):
        total_input = torch.cat((z,l), dim=dim)
        t = self.decoding_layers(total_input)
        return t

class cVAE(nn.Module):
    def __init__(self, Encoder, Decoder, latent_dim=6):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x, l):
        mean, logvar = self.Encoder(x, l)
        z = self.reparameterize(mean, torch.exp(.5*logvar))
        t = self.Decoder(z, l)
        return t, mean, logvar

class DetectorDataset(Dataset):
    def __init__(self, counts, L, p, x, y):
        self.counts = counts
        self.L = L
        self.p = p
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.counts)
    
    def __getitem__(self, index):
        sample = {
            "counts" : self.counts[index],
            "labels" : torch.tensor((
                self.L[index],
                self.p[index],
                self.x[index],
                self.y[index]
            ), dtype=torch.float)
        }

        return sample

class Scaler():
    def __init__(self, data_names, data_min_values, data_max_values):
        data_values = []
        for data_min_value, data_max_value in zip(data_min_values, data_max_values):
            data_values.append({"min":data_min_value, "max":data_max_value})
        self.scaling_data = dict(zip(data_names, data_values))
    
    def _get_min_max(self, data_name):
        return self.scaling_data[data_name]["min"], self.scaling_data[data_name]["max"]

    def downscale(self, data_name, data):
        min_value, max_value = self._get_min_max(data_name)
        downscaled_data = (data - min_value)/(max_value - min_value)
        return downscaled_data
    
    def upscale(self, data_name, downscaled_data):
        min_value, max_value = self._get_min_max(data_name)
        upscaled_data = min_value + downscaled_data * (max_value - min_value)
        return upscaled_data

def loss_function(args, output, target_counts, target_labels):
    t, mu, logvar = output

    mse_loss = nn.functional.mse_loss(t, torch.cat((target_counts, target_labels), dim=1))
    kld_loss = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + args.beta * kld_loss

def train(args, model, loader, optimizer, epoch, no_print=False):
    initial_time = time()
    model.train()
    train_losses = []
    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad()
        output = model(batch["counts"], batch["labels"])
        loss = loss_function(args, output, batch["counts"], batch["labels"])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if not no_print and batch_index % args.logging_interval == 0:
            print(f"Training epoch {epoch} [{batch_index*len(batch["counts"])}/{len(loader.dataset)}]\tLoss: {loss.item():.6f}", end="\r")
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Training epoch {epoch} complete\tAverage loss: {np.mean(train_losses):.6f}\t({time()-initial_time:.2f}s elapsed)")
    return train_losses

def test(args, model, loader, no_print=False):
    initial_time = time()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            output = model(batch["counts"], batch["labels"])
            losses.append(loss_function(args, output, batch["counts"], batch["labels"]).item())
            if not no_print:
                print(f"Testing progress [{batch_index*len(batch["counts"])}/{len(loader.dataset)}] {100.*batch_index*len(batch["counts"])/len(loader.dataset):.2f}%", end="\r")
    test_loss = np.mean(losses)
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Testing\t\t\t\tAverage loss: {test_loss:.6f}\t({time()-initial_time:.2f}s elapsed)")
    return test_loss

def paint_losses(args, train_losses, test_losses, train_loader, test_loader):
    fig, ax = plt.subplots()

    train_spots = np.arange(1, 1+args.epochs*(1+(len(train_loader.dataset)//args.train_batch_size))) * args.train_batch_size
    test_spots = np.arange(1, 1+len(test_losses)) * len(train_loader.dataset)

    ax.plot(train_spots[:len(train_losses)], train_losses, ls="solid", color="tab:blue", label="Train loss")
    ax.plot(test_spots, test_losses, ls="None", marker="o", color="tab:orange", label="Test loss")

    ax.xaxis.set_major_locator(plt.MultipleLocator(len(train_loader.dataset)))
    
    ax.set_xlim(left=0)

    ax.legend(loc="best")
    
    fig.savefig(f"vae/{args.folder_name}_figs/{args.save_name}_losses.pdf", dpi=300, bbox_inches="tight")

def generate_ax(args, model, ax, spots, counts, labels, minor=False):
    ax.plot(spots, counts, color = "tab:orange")

    """
    title_string, label_names = "", ("L", "p", "x", "y")
    for (label, name) in zip(labels, label_names):
        title_string += f" {name}={label} "
    ax.set_title(title_string)
    """

    ax.set_xlim(left=1, right=33*4)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=33))
    if minor:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

def generate_graphs(args, scaler, model, number=1):
    fig, axs = plt.subplots(number, number, figsize=(1.5*number,1.5*number), sharex="all", layout="compressed")

    spots = np.arange(1, 1+33*4)

    model.eval()
    with torch.no_grad():
        zs = torch.randn(args.print_number**2, model.latent_dim)
        Ls = 5 + 45 * np.random.rand(args.print_number**2)
        ps = 50 * np.random.rand(args.print_number**2)
        xs = -4000 + 8000 * np.random.rand(args.print_number**2)
        ys = -4000 + 8000 * np.random.rand(args.print_number**2)
        labs = torch.tensor((Ls,ps,xs,ys), dtype=torch.float).permute(1,0)
        t = model.Decoder(zs, labs, dim=1)

    if number == 1:
        t = t[0]
        counts, labels = t[:args.n_det], t[args.n_det:]
        counts = scaler.upscale("counts", counts)
        generate_ax(args, model, axs, spots, counts, labels, minor=True)
    else:
        for (ax, t_ax) in zip(axs.ravel(), t):
            counts_ax, labels_ax = t_ax[:args.n_det], t_ax[args.n_det:]
            counts_ax = scaler.upscale("counts", counts_ax)
            generate_ax(args, model, ax, spots, counts_ax, labels_ax, minor=False)

    fig.supxlabel("Detector position")
    fig.supylabel("Photon count")

    fig.savefig(f"vae/{args.folder_name}_gen/{args.save_name}_gen.pdf", dpi=300, bbox_inches="tight")

def compare_ax(args, model, ax, spots, true_counts, true_labels, gen_counts, gen_labels, minor=False):

    ax.plot(spots, true_counts, ls="solid", color = "tab:blue")
    ax.plot(spots, gen_counts, ls="dashed", color = "tab:orange")

    ax.set_xlim(left=1, right=33*4)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=33))
    if minor:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Photon count")

def compare_graph(args, scaler, model, true):
    fig, ax = plt.subplots(figsize=(5,5))

    spots = np.arange(1, 1+33*4)

    L_truth = true["Ls"] == np.unique(true["Ls"])[args.gen_values[0]]
    p_truth = true["ps"] == np.unique(true["ps"])[args.gen_values[1]]
    x_truth = true["xs"] == np.unique(true["xs"])[args.gen_values[2]]
    y_truth = true["ys"] == np.unique(true["ys"])[args.gen_values[3]]

    index = np.where(L_truth & p_truth & x_truth & y_truth)[0][0]

    true_counts = true["counts"][index]
    true_L = true["Ls"][index].item()
    true_p = true["ps"][index].item()
    true_x = true["xs"][index].item()
    true_y = true["ys"][index].item()
    true_labels = (true_L, true_p, true_x, true_y)
    true_L_down = scaler.downscale("L", true_L)
    true_p_down = scaler.downscale("p", true_p)
    true_x_down = scaler.downscale("x", true_x)
    true_y_down = scaler.downscale("y", true_y)
    true_labels_down = (true_L_down, true_p_down, true_x_down, true_y_down)
    true_labels_down_tensor = torch.tensor(true_labels_down, dtype=torch.float).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        zs = torch.randn(1, model.latent_dim)
        t = model.Decoder(zs, true_labels_down_tensor, dim=1)[0]

    gen_counts, gen_labels = t[:args.n_det], t[args.n_det:]
    
    upscaled_gen_counts = scaler.upscale("counts", gen_counts)

    upscaled_gen_labels = []
    label_names = list(scaler.scaling_data.keys())[1:]
    for gen_label, label_name in zip(gen_labels, label_names):
        upscaled_gen_labels.append(scaler.upscale(label_name, gen_label))

    compare_ax(args, model, ax, spots, true_counts, true_labels, upscaled_gen_counts, upscaled_gen_labels, minor=True)

    true_title, gen_title = "", ""
    for (true_l, gen_l, name) in zip(true_labels, upscaled_gen_labels, label_names):
        true_title += f" {name}={true_l:.2f} "
        gen_title += f" {name}={gen_l:.2f} "
    
    ax.set_title(gen_title, color="tab:orange")
    fig.suptitle(true_title, color="tab:blue")

    fig.savefig(f"vae/{args.folder_name}_gen/{args.save_name}_comp.pdf", dpi=300, bbox_inches="tight")

def create_scaler():
    data_names = ("counts", "L", "p", "x", "y")
    data_min_values = (0., 5., 10., -4000., -4000.)
    data_max_values = (905., 50., 50., 4000., 4000.)
    return Scaler(data_names, data_min_values, data_max_values)

def create_parser():
    parser = argparse.ArgumentParser(description="Conditional Variational Autoencoder")
    
    # File settings
    parser.add_argument("--folder-name", type=str, default="vae",
                        help="input folder for storing figures (default=vae)")
    parser.add_argument("--save-name", type=str, default="vae",
                        help="input file save name (default=vae)")
    parser.add_argument("--save-path", type=str, default="vae/vae_models",
                        help="input model save path (default=vae/vae_models)")

    # Training settings
    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="input training batch size (default=64)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="input testing batch size (default=1000)")
    parser.add_argument("--element-cutoff", type=int, default=5000,
                        help="input dataset element cutoff (default=5000)") # sufficient memory -> default=-1
    parser.add_argument("--epochs", type=int, default=100,
                        help="input number of epochs (default=100)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="input learning rate (default=0.001)")
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
    parser.add_argument("--print-type", type=str, default="gen",
                        help="print generated graphs (default, gen) or comparison (comp)")
    parser.add_argument("--latent-dim", type=int, default=3,
                        help="input latent space dimensionality (default=8)")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="input hidden layers dimensionality (default=32)")
    parser.add_argument("--beta", type=float, default=.5,
                        help="input kldiv loss weight (default=0.5)")
    parser.add_argument("--n-det", type=int, default=132)
    parser.add_argument("--n-lab", type=int, default=4)
    parser.add_argument("--gen-values", type=int, nargs="+",
                        help="input (L,p,x,y) values for generated output, as an index on the unique list")
    
    return parser

def reconstruct_avalanche(args, counts_list):
    """
    Expects counts_list to be ordered as (x1, x2, y1, y2).
    """

    Ns, means, devs = [], [], []
    for counts in counts_list:
        Ns.append(torch.sum(counts))
        means.append(torch.mean(counts))
        devs.append(np.sqrt(torch.var(counts)))
    
    nd = lambda i : Ns[i]/devs[i]
    avalanche_eq = lambda i, j : ((means[i]*nd(i))+(means[j]*nd(j)))/(nd(i)+nd(j))

    x_avalanche, y_avalanche = avalanche_eq(0,1), avalanche_eq(2,3)
    return x_avalanche, y_avalanche

def save_model(args, model, scaler):

    full_path = f"{args.save_path}/{args.save_name}"

    Path(f"{full_path}").mkdir(parents=True, exist_ok=True)

    save_data = {
        "args" : args,
        "scaler" : scaler
    }

    with open(f"{full_path}/{args.save_name}_data", "wb") as handle:
        pickle.dump(save_data, handle)
    torch.save(model.state_dict(), f"{full_path}/{args.save_name}_model.pt")

def main():
    main_time = time()
    print("Program start")
    
    parser = create_parser()
    args = parser.parse_args()

    scaler = create_scaler()

    torch.manual_seed(args.seed)

    enc = Encoder(input_dim=args.n_det+args.n_lab, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    dec = Decoder(latent_dim=args.latent_dim+args.n_lab, hidden_dim=args.hidden_dim, output_dim=args.n_det+args.n_lab)
    model = cVAE(Encoder=enc, Decoder=dec, latent_dim=args.latent_dim)
    if not args.load:
        df = pd.DataFrame(pd.read_pickle("vae/simulated_events.pickle"))
        photon_counts = torch.tensor(list(df["nphotons"].values)[:args.element_cutoff], dtype=torch.float)
        Ls = torch.tensor(list(df["dcol"].values)[:args.element_cutoff], dtype=torch.float)
        ps = torch.tensor(list(df["p"].values)[:args.element_cutoff], dtype=torch.float)
        xs = torch.tensor(list(df["x"].values)[:args.element_cutoff], dtype=torch.float)
        ys = torch.tensor(list(df["y"].values)[:args.element_cutoff], dtype=torch.float)

        photon_counts = scaler.downscale("counts", photon_counts)
        Ls = scaler.downscale("L", Ls)
        ps = scaler.downscale("p", ps)
        xs = scaler.downscale("x", xs)
        ys = scaler.downscale("y", ys)

        train_counts, test_counts = train_test_split(photon_counts, test_size=.2, random_state=args.seed)
        train_Ls, test_Ls = train_test_split(Ls, test_size=.2, random_state=args.seed)
        train_ps, test_ps = train_test_split(ps, test_size=.2, random_state=args.seed)
        train_xs, test_xs = train_test_split(xs, test_size=.2, random_state=args.seed)
        train_ys, test_ys = train_test_split(ys, test_size=.2, random_state=args.seed)

        train_loader = DataLoader(
            DetectorDataset(train_counts, train_Ls, train_ps, train_xs, train_ys),
            batch_size=args.train_batch_size, shuffle=True
        )

        test_loader = DataLoader(
            DetectorDataset(test_counts, test_Ls, test_ps, test_xs, test_ys),
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
        for epoch in range(1, 1+args.epochs):
            train_log = train(args, model, train_loader, optimizer, epoch)
            test_loss = test(args, model, test_loader)

            train_losses.append(train_log)
            test_losses.append(test_loss)
        print("Training end")

        if args.save:
            save_model(args, model, scaler)
            print("Model saved")
        
        paint_losses(args, np.ravel(train_losses), test_losses, train_loader, test_loader)
        print("Losses painted")

    else:
        model.load_state_dict(torch.load(f"vae/{args.folder_name}_models/{args.save_name}_model.pt"))
        model.eval()
        print("Model loaded")

    if args.print_type == "gen":
        generate_graphs(args, scaler, model, number=args.print_number)
    elif args.print_type == "comp":
        df = pd.DataFrame(pd.read_pickle("data/simulated_events.pickle"))
        photon_counts = torch.tensor(list(df["nphotons"].values)[:args.element_cutoff], dtype=torch.float)
        Ls = torch.tensor(list(df["dcol"].values)[:args.element_cutoff], dtype=torch.float)
        ps = torch.tensor(list(df["p"].values)[:args.element_cutoff], dtype=torch.float)
        xs = torch.tensor(list(df["x"].values)[:args.element_cutoff], dtype=torch.float)
        ys = torch.tensor(list(df["y"].values)[:args.element_cutoff], dtype=torch.float)
        true = {
            "counts" : photon_counts,
            "Ls" : Ls,
            "ps" : ps,
            "xs" : xs,
            "ys" : ys
        }
        print("Data loaded")
        compare_graph(args, scaler, model, true)
    print("Graphs generated")

    main_total = time()-main_time
    main_days = int(main_total // 86400)
    main_hours = int((main_total % 86400) // 3600)
    main_minutes = int(((main_total % 86400) % 3600) // 60)
    main_seconds = int(((main_total % 86400) % 3600) % 60)
    print(f"Program complete ({main_days:02}:{main_hours:02}:{main_minutes:02}:{main_seconds:02} elapsed)")

if __name__ == "__main__":
    main()
