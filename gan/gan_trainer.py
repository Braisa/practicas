import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
import pickle
from pathlib import Path
from time import time
from gan_models import create_GAN, Discriminator
from utils import counter_dataset
from utils.early_stopper import EarlyStopper

def wasserstein_gp_loss(args, discriminator: Discriminator, real, fake, gp_lambda: float):
    """
    Computes the Wasserstein loss with gradient penalty for the GAN.

    Parameters
    ----------
    discriminator : Discriminator
        Discriminator model of the GAN.
    real
        Set of real inputs that will go through the Discriminator.
    fake
        Set of fake outputs produced by the Generator.
    gp_lambda : float
        Hyperparameter that controls the weight of the gradient penalty.
    
    Returns
    -------
    total_loss
        Weighted sum of the Wasserstein loss computed between real and fake,
        and the gradient penalty weighted by gp_lambda.
    """
    interpol_weight = torch.Tensor(np.random.random((real.size(), 1, 1, 1)))
    interpols = (interpol_weight * real + ((1-interpol_weight) * fake)).requires_grad_(True).to(args.device)
    fake_var = torch.Tensor(real.shape(), 1).fill_(1.).to(args.device)
    gradients = autograd.grad(
        outputs=discriminator(interpols),
        inputs=interpols,
        grad_outputs=fake_var,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0].to(args.device)
    gradients = gradients.view(gradients.size(0), -1).to(args.device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    real_val, fake_val = discriminator(real), discriminator(fake)
    w_loss = torch.mean(fake_val) - torch.mean(real_val)

    total_loss = w_loss + gp_lambda * gradient_penalty
    return total_loss

def _print_progress(header, current, max, message):
    print(f"{header}\t[{current}/{max}]\t[{100*current/max:.1f}%]\t{message}", end="\r")

def _print_finish(header, message):
    print(f"\033[K", end="\r")
    print(f"{header}\t{message}")

def _pretty_time(time):
    days = int(time // 86400)
    hours = int((time % 86400) // 3600)
    minutes = int(((time % 86400) % 3600) // 60)
    seconds = int(((time % 86400) % 3600) % 60)
    time_string = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
    return time_string

def train_epoch(args, generator, discriminator, loader, gen_optimizer, dis_optimizer, epoch, no_print=False):
    initial_time = time()
    generator.train()
    discriminator.train()
    gen_losses, dis_losses = [], []
    for batch_index, batch in enumerate(loader):
        counts, labels = batch["counts"].to(args.device), batch["labels"].to(args.device)
        real = torch.cat((counts, labels), dim=1).to(args.device)

        dis_optimizer.zero_grad()
        z = torch.tensor(np.random.normal(0, 1, (args.nonlabel_input_dim+args.label_input_dim, args.latent_dim))).to(args.device)
        fake = generator(z, labels)
        fake_counts, fake_labels = fake[:args.nonlabel_input_dim], fake[args.nonlabel_input_dim:]
        real_val, fake_val = discriminator(counts, labels), discriminator(fake_counts, fake_labels)
        dis_loss = wasserstein_gp_loss(args, discriminator, real, fake, args.gp_lambda)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        gen_train = batch_index % args.critique == 0
        gen_optimizer.zero_grad()
        if gen_train:
            gen_loss = -1 * torch.mean(fake_val)
            gen_loss.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss.item())
        
            if args.print_progress:
                header = f"TRAINING\tEpoch {epoch}"
                current, max = batch_index*len(batch["counts"]), len(loader.dataset)
                message = f"Dis Loss: {dis_loss.item():.6f}\tGen Loss: {gen_loss.item():.6f}"
                if not no_print: _print_progress(header, current, max, message)
    
    gen_train_loss, dis_train_loss = np.mean(gen_losses), np.mean(dis_losses)

    if args.print_progress:
        header = f"TRAINING\tEpoch {epoch} end"
        message = f"Average Dis Loss: {dis_train_loss:.6f}\tAverage Gen Loss: {gen_train_loss:.6f}\t({_pretty_time(time()-initial_time)} elapsed)"
        if not no_print: _print_finish(header, message)
    
    return gen_train_loss, dis_train_loss

def test(args, generator, discriminator, loader, epoch=None, early_stopper=None, no_print=False):
    initial_time = time()
    generator.eval()
    discriminator.eval()
    gen_losses, dis_losses = [], []
    early_stop = False
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            counts, labels = batch["counts"].to(args.device), batch["labels"].to(args.device)
            real = torch.cat((counts, labels), dim=1).to(args.device)

            z = torch.tensor(np.random.normal(0, 1, (args.nonlabel_input_dim+args.label_input_dim, args.latent_dim))).to(args.device)
            fake = generator(z, labels)
            fake_counts, fake_labels = fake[:args.nonlabel_input_dim], fake[args.nonlabel_input_dim:]
            real_val, fake_val = discriminator(counts, labels), discriminator(fake_counts, fake_labels)
            dis_loss = wasserstein_gp_loss(args, discriminator, real, fake, args.gp_lambda)
            dis_losses.append(dis_loss.item())

            gen_test = batch_index % args.critique == 0
            if gen_test:
                gen_loss = -1 * torch.mean(fake_val)
                gen_losses.append(gen_loss.item())

                if args.print_progress:
                    if not epoch: header = f" TESTING"
                    else: header = f" TESTING\tEpoch {epoch}"
                    current, max = batch_index*len(batch["counts"]), len(loader.dataset)
                    message = f"Dis Loss: {dis_loss.item():.6f}\tGen Loss: {gen_loss.item():.6f}"
                    if not no_print: _print_progress(header, current, max, message)
    
    gen_test_loss, dis_test_loss = np.mean(gen_losses), np.mean(dis_losses)
    if args.print_progress:
        if not epoch: header = f" TESTING\tend"
        else: header = f" TESTING\t Epoch {epoch} end"
        message = f"Average Dis Loss: {np.mean(dis_test_loss):.6f}\tAverage Gen Loss: {np.mean(gen_losses):.6f}\t({_pretty_time(time()-initial_time)} elapsed)"
        if not no_print: _print_finish(header, message)
    
    if early_stopper: early_stop = early_stopper.check_early_stop(gen_test_loss)

    return gen_test_loss, dis_test_loss, early_stop

def train_for_epochs(args, generator, discriminator, gen_optimizer, dis_optimizer, train_loader, test_loader, epochs, early_stopper=None, no_print=False):
    gen_train_losses, dis_train_losses = [], []
    gen_test_losses, dis_test_losses = [], []

    if args.print_progress: print("Training start")
    for epoch in range(1, 1+epochs):
        gen_train_loss, dis_train_loss = train_epoch(args, generator, discriminator, train_loader, gen_optimizer, dis_optimizer, epoch, no_print=no_print)
        gen_train_losses.append(gen_train_loss)
        dis_train_losses.append(dis_train_loss)
        gen_test_loss, dis_test_loss, early_stop = test(args, generator, discriminator, test_loader, epoch=epoch, early_stopper=early_stopper, no_print=no_print)
        gen_test_losses.append(gen_test_loss)
        dis_test_losses.append(dis_test_loss)
        if early_stop:
            if args.print_progress: print(f"Early stop at epoch {epoch}.")
            break
    if args.print_progress: print("Training end")

    return gen_train_losses, gen_test_losses, dis_train_losses, dis_test_losses

def get_datasets(args, scaler):
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
    if not args.return_full_data:
        return train_dataset, test_dataset, None
    else:
        full_data = {"counts" : counts, "L" : Ls, "p" : ps, "x" : xs, "y" : ys}
        return train_dataset, test_dataset, full_data

def initialise_data(args):
    scaler = counter_dataset.create_counter_scaler()
    train_dataset, test_dataset, full_data = get_datasets(args, scaler)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    return scaler, train_loader, test_loader, full_data

def get_args():
    parser = argparse.ArgumentParser(description="WGAN-GP Trainer")
    # File settings
    parser.add_argument("savename", type=str,
                        help="Save name")
    parser.add_argument("--path", type=str, default="gan/saves",
                        help="Parent folder path (default=gan/saves)")
    # Model settings
    parser.add_argument("--nonlabel-input-dim", "-nlid", type=int, default=132,
                        help="Non-label input size (default=132)")
    parser.add_argument("--label-input-dim", "-lid", type=int, default=4,
                        help="Label input size (default=4)")
    parser.add_argument("--generator-hidden-dim", "-ghd", type=int, default=32,
                        help="Generator hidden layer size (default=32)")
    parser.add_argument("--discriminator-big-hidden-dim", "-bhd", type=int, default=64,
                        help="Discriminator big hidden layer size (default=64)")
    parser.add_argument("--discriminator-small-hidden-dim", "-shd", type=int, default=16,
                        help="Discriminator small hidden layer size (default=16)")
    parser.add_argument("--generator-hidden-layers", "-ghl", type=int, default=2,
                        help="Generator hidden layer amount (default=2)")
    parser.add_argument("--discriminator-big-hidden-layers", "-bhl", type=int, default=2,
                        help="Discriminator big hidden layer amount (default=2)")
    parser.add_argument("--discriminator-small-hidden-layers", "-shl", type=int, default=2,
                        help="Discriminator small hidden layer amount (default=2)")
    parser.add_argument("--latent-dim", "-ld", type=int, default=3,
                        help="Generator latent space size (default=3)")
    parser.add_argument("--gp-lambda", "-lam", type=float, default=50.0,
                        help="Gradient penalty weight in Wasserstein loss (default=50.0)")
    # Training settings
    parser.add_argument("--data-path", type="str", default="data/simulated_events.pickle",
                        help="Full path to pickle containing the training data (default=data/simulated_events.pickle)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test set split size (default=0.2)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default=50)")
    parser.add_argument("--generator-learning-rate", "-glr", type=float, default=1e-3,
                        help="Generator optimizer learning rate (default=1e-3)")
    parser.add_argument("--discriminator-learning-rate", "-dlr", type=float, default=1e-3,
                        help="Discriminator optimizer learning rate (default=1e-3)")
    parser.add_argument("--critique", type=int, default=5,
                        help="Number of discriminator train batches for each generator train batch (default=5)")
    parser.add_argument("--train-batch-size", type=int, default=512,
                        help="Training batch size (default=512)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="Testing batch size (default=1000)")
    parser.add_argument("--early-stop", type=bool, default=True,
                        help="Whether to use an early stopper (default=True)")
    parser.add_argument("--early-stop-rel-delta", type=float, default=0.1,
                        help="Relative delta for the early stopper (default=0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Patience for the early stopper (default=5)")
    # Miscellaneous
    parser.add_argument("--disable-cuda", type=bool, default=False,
                        help="Whether to disable CUDA (default=False)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for torch (default=1)")
    parser.add_argument("--print-progress", type=bool, default=True,
                        help="Whether to print program progress or not (default=True)")
    parser.add_argument("--epoch-print", type=bool, default=True,
                        help="Whether to print epoch progress or not (default=True)")
    parser.add_argument("--return-full-data", type=bool, default=False,
                        help="Whether to store full data on its own (default=False)")
    args = parser.parse_args()
    args.device = torch.device("cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu")
    return args

def main():
    main_time = time()
    print("Program start")

    args = get_args()
    scaler, train_loader, test_loader, _ = initialise_data(args)
    if args.print_progress: print("Data initialised")

    if not args.early_stop: early_stopper = None
    else: early_stopper = EarlyStopper(patience=args.early_stop_patience, rel_delta=args.early_stop_rel_delta)

    torch.manual_seed(args.seed)

    generator, discriminator = create_GAN(
        nonlabel_input_dim=args.nonlabel_input_dim,
        label_input_dim=args.label_input_dim,
        gen_hidden_dim=args.generator_hidden_dim,
        big_hidden_dim=args.discriminator_big_hidden_dim,
        small_hidden_dim=args.discriminator_small_hidden_dim,
        latent_dim=args.latent_dim,
        gen_hidden_layers=args.generator_hidden_layers,
        dis_hidden_layers=args.discriminator_hidden_layers
    )
    generator, discriminator = generator.to(args.device), discriminator.to(args.device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_learning_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate)

    gen_train_losses, gen_test_losses, dis_train_losses, dis_test_losses = train_for_epochs(args, generator, discriminator, gen_optimizer, dis_optimizer, train_loader, test_loader, args.epochs, early_stopper=early_stopper, no_print=not args.epoch_print)
    if args.print_progress: print("GAN trained")

    print(f"Program end ({_pretty_time(time()-main_time)} elapsed)")

if __name__ == "__main__":
    main()
