import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import argparse
import pickle
from pathlib import Path
import pandas as pd
import sys
sys.path.append("/scratch12/brais_otero/practicas/")
from cvae import create_cVAE
from cvae_trainer import train_for_epochs, get_datasets
from utils.counter_dataset import create_counter_scaler
from time import time
from utils.early_stopper import EarlyStopper

def objective(args, trial, train_loader, test_loader, early_stopper=None):
    hidden_dim_trial = trial.suggest_int("hidden_dim", 2, 128)
    hidden_layers_trial = trial.suggest_int("hidden_layers", 2, 8)
    lr_trial = trial.suggest_float("learning_rate", 1e-5, 1e-1)

    nlid, lid, ld = args.nonlabel_input_dim, args.label_input_dim, args.latent_dim
    model = create_cVAE(
        nonlabel_input_dim=nlid,
        label_input_dim=lid,
        hidden_dim=hidden_dim_trial,
        latent_dim=ld,
        enc_hidden_layers=hidden_layers_trial,
        dec_hidden_layers=hidden_layers_trial
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_trial)

    train_losses, test_losses = train_for_epochs(args, model, optimizer, train_loader, test_loader, args.trial_epochs, early_stopper=early_stopper)
    return test_losses[-1]

def _pretty_time(time):
    days = int(time // 86400)
    hours = int((time % 86400) // 3600)
    minutes = int(((time % 86400) % 3600) // 60)
    seconds = int(((time % 86400) % 3600) % 60)
    time_string = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
    return time_string

def print_best(study, trial, initial_time):
    best_trial = study.best_trial
    print(f"Current trial: {trial.number} | Hitherto best: {best_trial.number}")
    print(f"Current value: {trial.value} | Hitherto best: {best_trial.value}")
    print(f"Current params: {trial.params} | Hitherto best: {best_trial.params}")
    print(f"Time elapsed: {_pretty_time(time()-initial_time)}")

def save_best(args, study):
    full_path = f"{args.path}/{args.savename}/optuna"
    Path(f"{full_path}").mkdir(parents=True, exist_ok=True)
    with open(f"{full_path}/{args.savename}_best_params.pickle", "wb") as handle:
        pickle.dump(study.best_params, handle)

def get_args():
    parser = argparse.ArgumentParser(description="cVAE Trainer")
    # Study settings
    parser.add_argument("savename", type=str, default="test",
                        help="Study save name (default=test)")
    parser.add_argument("trials", type=int, default=1000,
                        help="Number of study trials (default=1000)")
    parser.add_argument("--trial-epochs", type=int, default=50,
                        help="Number of epochs for each trial (default=50)")
    parser.add_argument("--path", type=str, default="cvae/saves",
                        help="Parent folder path (default=cvae/saves)")
    
    # Model settings
    parser.add_argument("--nonlabel-input-dim", "-nlid", type=int, default=132,
                        help="Non-label input size (default=132)")
    parser.add_argument("--label-input-dim", "-lid", type=int, default=4,
                        help="Label input size (default=4)")
    parser.add_argument("--latent-dim", "-ld", type=int, default=3,
                        help="Latent space size (default=3)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Latent shape loss weight (default=0.5)")
    # Training settings
    parser.add_argument("--data-path", type=str, default="data/simulated_events.pickle",
                        help="Full path to pickle containing the training data (default=data/simulated_events.pickle)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test set split size (default=0.2)")
    parser.add_argument("--train-batch-size", type=int, default=512,
                        help="Training batch size (default=512)")
    parser.add_argument("--test-batch-size", type=str, default=1000,
                        help="Testing batch size (default=1000)")
    parser.add_argument("--early-stop", type=bool, default=True,
                        help="Whether to use an early stopper (default=True)")
    parser.add_argument("--early-stop-rel-delta", type=float, default=0.05,
                        help="Relative delta for the early stopper (default=0.05)")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Patience for the early stopper (default=10)")
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

def main():
    main_time = time()

    args = get_args()

    train_dataset, test_dataset, full_data = get_datasets(args, create_counter_scaler())
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    if not args.early_stop: early_stopper = None
    else: early_stopper = EarlyStopper(patience=args.early_stop_patience, rel_delta=args.early_stop_rel_delta)
    
    torch.manual_seed(args.seed)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(lambda trial : objective(args, trial, train_loader, test_loader, early_stopper=early_stopper), n_trials=args.trials, callbacks=[lambda study, trial : print_best(study, trial, main_time)])
    
    best_trial = study.best_trial
    print(f"----------\nBest trial ({best_trial.number})\n----------")
    print(f"Value: {best_trial.value}")
    print(f"Params: {best_trial.params}")
    print(f"Total time elapsed: {_pretty_time(time()-main_time)}")

    save_best(args, study)

if __name__ == "__main__":
    main()
