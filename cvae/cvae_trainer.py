import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cvae
import counter_dataset
import argparse
import pickle
from pathlib import Path
from time import time

def loss_function(output, target_counts, target_labels, beta):
    """
    Computes the loss functions for a Conditional Variational Autoencoder. This is
    comprised in part by a reconstruction loss (in this case, a mean squared error
    between input and output) and in part by a latent shape loss (in this case, a
    Kullback-Leibler divergence for a normalized gaussian), the latter of which is
    weighted by a hyperparameter beta.

    Parameters
    ----------
    output
        Output from a cVAE. This includes both the neuron outputs on the final layer
        of the Decoder as well as the mean and log-variance that was encoded onto the
        latent space.
    target_counts
        Photon counts against which the reconstruction loss will be computed, in
        combination with the target labels.
    target_labels
        Collimation length, gas pressure and ionizing particle position against which
        the reconstruction loss will be computed, in combination with the target counts.
    beta : float
        Hyperparameter that controls the weight of the latent shape loss.
    
    Returns
    -------
    total_loss
        Weighted sum of the reconstruction and latent shape losses.
    """
    t, mu, logvar = output
    mse_loss = nn.functional.mse_loss(t, torch.cat((target_counts, target_labels), dim=1))
    kld_loss = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = mse_loss + beta * kld_loss
    return total_loss

def _print_progress(header, current, max, message):
    print(f"{header}\t[{current}/{max}]\t[{100*current/max:.1f}%]\t{message}", end="\r")

def _print_finish(header, message):
    print(f"\033[K", end="\r")
    print(f"{header}\t{message}")

def train_epoch(args, model, loader, optimizer, epoch, no_print=False):
    initial_time = time()
    model.train()
    train_losses = []
    for batch_index, batch in enumerate(loader):
        counts, labels = batch["counts"].to(args.device), batch["labels"].to(args.device)
        optimizer.zero_grad()
        output = model(counts, labels)
        loss = loss_function(output, counts, labels, args.beta)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach().cpu().numpy())
        if args.print_progress and batch_index % args.logging_interval == 0:
            header = f"TRAINING\tEpoch {epoch}"
            current, max = batch_index*len(batch["counts"]), len(loader.dataset)
            message = f"Loss: {loss.item():.6f}"
            if not no_print:
                _print_progress(header, current, max, message)
    if args.print_progress and not no_print:
        header = f"TRAINING\tEpoch {epoch} end"
        message = f"Average loss: {np.mean(train_losses):.6f}\t({time()-initial_time:.2f}s elapsed)"
        _print_finish(header, message)
    return train_losses

def test(args, model, loader, epoch=None, no_print=False):
    initial_time = time()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            counts, labels = batch["counts"].to(args.device), batch["labels"].to(args.device)
            output = model(counts, labels)
            current_loss = loss_function(output, counts, labels, args.beta).detach().cpu().numpy()
            losses.append(current_loss)
            if args.print_progress:
                if not epoch:
                    header = f" TESTING"
                else:
                    header = f" TESTING\tEpoch {epoch}"
                current, max = batch_index*len(batch["counts"]), len(loader.dataset)
                message = f"Loss: {current_loss:.6f}"
                if not no_print:
                    _print_progress(header, current, max, message)
    test_loss = np.mean(losses)
    if args.print_progress and not no_print:
        if not epoch:
            header = f" TESTING\tend"
        else:
            header = f" TESTING\tEpoch {epoch} end"
        message = f"Average loss: {test_loss:.6f}\t({time()-initial_time:.2f}s elapsed)"
        _print_finish(header, message)
    return test_loss

def get_args():
    parser = argparse.ArgumentParser(description="cVAE Trainer")
    # File settings
    parser.add_argument("savename", type=str, default="test",
                        help="Save name (default=test)")
    parser.add_argument("--path", type=str, default="cvae/saves",
                        help="Parent folder path (default=cvae/saves)")
    # Model settings
    parser.add_argument("--nonlabel-input-dim", "-nlid", type=int, default=132,
                        help="Non-label input size (default=132)")
    parser.add_argument("--label-input-dim", "-lid", type=int, default=4,
                        help="Label input size (default=4)")
    parser.add_argument("--hidden-dim", "-hd", type=int, default=32,
                        help="Hidden layer size (default=32)")
    parser.add_argument("--latent-dim", "-ld", type=int, default=3,
                        help="Latent space size (default=3)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Latent shape loss weight (default=0.5)")
    # Training settings
    parser.add_argument("--data-path", type=str, default="data/simulated_events.pickle",
                        help="Full path to pickle containing the training data (default=data/simulated_events.pickle)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test set split size (default=0.2)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default=50)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                        help="Optimizer learning rate (default=1e-3)")
    parser.add_argument("--train-batch-size", type=int, default=512,
                        help="Training batch size (default=512)")
    parser.add_argument("--test-batch-size", type=str, default=1000,
                        help="Testing batch size (default=1000)")
    # Miscellaneous
    parser.add_argument("--disable-cuda", type=bool, default=False,
                        help="Whether to disable CUDA (default=False)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for torch (default=1)")
    parser.add_argument("--logging-interval", type=int, default=50,
                        help="Training log print interval (default=50)")
    parser.add_argument("--print-progress", type=bool, default=True,
                        help="Whether to print epoch progress or not (default=True)")
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    return args

def get_datasets(args, scaler, return_full_data=True):
    df = pd.DataFrame(pd.read_pickle(args.data_path))
    counts = scaler.downscale("counts", torch.tensor(list(df["nphotons"].values), dtype=torch.float))
    Ls = scaler.downscale("L", torch.tensor(list(df["dcol"].values), dtype=torch.float))
    ps = scaler.downscale("p", torch.tensor(list(df["p"].values), dtype=torch.float))
    xs = scaler.downscale("x", torch.tensor(list(df["x"].values), dtype=torch.float))
    ys = scaler.downscale("y", torch.tensor(list(df["y"].values), dtype=torch.float))
    
    tr_c, te_c = train_test_split(counts, test_size=args.test_split, random_state=args.seed)
    tr_l, te_l = train_test_split(Ls, test_size=args.test_split, random_state=args.seed)
    tr_p, te_p = train_test_split(ps, test_size=args.test_split, random_state=args.seed)
    tr_x, te_x = train_test_split(xs, test_size=args.test_split, random_state=args.seed)
    tr_y, te_y = train_test_split(ys, test_size=args.test_split, random_state=args.seed)

    train_dataset = counter_dataset.create_counter_dataset(tr_c, tr_l, tr_p, tr_x, tr_y)
    test_dataset = counter_dataset.create_counter_dataset(te_c, te_l, te_p, te_x, te_y)
    if not return_full_data:
        return train_dataset, test_dataset
    else:
        full_data = {"counts" : counts, "L" : Ls, "p" : ps, "x" : xs, "y" : ys}
        return train_dataset, test_dataset, full_data

def save_model(args, model, scaler, data, train_dataset, test_dataset, train_loader, test_loader, train_losses, test_losses):
    full_path = f"{args.path}/{args.savename}"
    Path(f"{full_path}").mkdir(parents=True, exist_ok=True)
    save_data = {
        "args" : args,
        "scaler" : scaler,
        "data" : data,
        "datasets" : {
            "train" : train_dataset,
            "test" : test_dataset
        },
        "loaders" : {
            "train" : train_loader,
            "test" : test_loader
        },
        "losses" : {
            "train" : train_losses,
            "test" : test_losses
        }
    }
    with open(f"{full_path}/{args.savename}_data.pickle", "wb") as handle:
        pickle.dump(save_data, handle)
    torch.save(model.state_dict(), f"{full_path}/{args.savename}_model.pt")

def train_for_epochs(args, model, optimizer, train_loader, test_loader, epochs, no_print=False):
    if args.print_progress: print("Training start")
    train_losses, test_losses = [], []
    for epoch in range(1, 1+epochs):
        train_log = train_epoch(args, model, train_loader, optimizer, epoch, no_print=no_print)
        train_losses.append(train_log)
        test_loss = test(args, model, test_loader, epoch=epoch, no_print=no_print)
        test_losses.append(test_loss)
    if args.print_progress: print("Training end")
    return train_log, test_losses

def main():
    main_time = time()
    print("Program start")

    args = get_args()
    scaler = counter_dataset.create_counter_scaler()
    
    torch.manual_seed(args.seed)

    enc = cvae.Encoder(input_dim=args.nonlabel_input_dim+args.label_input_dim, hidden_dim=args.hidden_dim, inner_dim=args.latent_dim)
    dec = cvae.Decoder(inner_dim=args.latent_dim+args.label_input_dim, hidden_dim=args.hidden_dim, output_dim=args.nonlabel_input_dim+args.label_input_dim)
    model = cvae.cVAE(Encoder=enc, Decoder=dec, latent_dim=args.latent_dim, label_dim=args.label_input_dim).to(args.device)

    train_dataset, test_dataset, full_data = get_datasets(args, scaler)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    if args.print_progress: print("Data loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_log, test_losses = train_for_epochs(args, model, optimizer, train_loader, test_loader, args.epochs)

    save_model(args, model, scaler, full_data, train_dataset, test_dataset, train_loader, test_loader, train_log, test_losses)
    if args.print_progress: print("Model saved")

    main_total = time()-main_time
    main_days = int(main_total // 86400)
    main_hours = int((main_total % 86400) // 3600)
    main_minutes = int(((main_total % 86400) % 3600) // 60)
    main_seconds = int(((main_total % 86400) % 3600) % 60)
    print(f"Program complete ({main_days:02}:{main_hours:02}:{main_minutes:02}:{main_seconds:02} elapsed)")

if __name__ == "__main__":
    main()
