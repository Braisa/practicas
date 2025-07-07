import torch # type: ignore
import torch.nn as nn # type: ignore
import optuna # type: ignore
import argparse
import pickle
from pathlib import Path
import pandas as pd # type: ignore
from cvae import Encoder, Decoder, cVAE
from cvae_trainer import train_for_epochs

def objective(args, trial):
    data = pd.read_pickle(f"{args.path}/{args.model_savename}/{args.model_savename}_data.pickle")
    margs = data["args"]

    train_loader, test_loader = data["loaders"]["train"], data["loaders"]["test"]

    hidden_dim_trial = trial.suggest_int("hidden_dim", 2, 128)
    lr_trial = trial.suggest_float("learning_rate", 1e-5, 1e-1)

    nlid, lid, ld = margs.nonlabel_input_dim, margs.label_input_dim, margs.latent_dim
    enc = Encoder(input_dim=nlid+lid, hidden_dim=hidden_dim_trial, inner_dim=ld)
    dec = Decoder(inner_dim=ld+lid, hidden_dim=hidden_dim_trial, output_dim=nlid+lid)
    model = cVAE(Encoder=enc, Decoder=dec, latent_dim=ld, label_dim=lid).to(args.device)
    optimizer = nn.optim.Adam(model.parameters(), lr=lr_trial)

    train_losses, test_losses = train_for_epochs(margs, model, optimizer, train_loader, test_loader, args.trial_epochs)
    return test_losses[-1]

def print_best(study, trial):
    print(f"Trial {trial.number}: {trial.params}")
    best_trial = study.best_trial
    print(f"Hitherto best value: {best_trial.value}")
    print(f"Hitherto best params: {best_trial.params}")

def save_best(args, study):
    full_path = f"{args.path}/{args.model_savename}/optuna"
    Path(f"{full_path}").mkdir(parents=True, exist_ok=True)
    with open(f"{full_path}/{args.study_savename}_best_params.pickle") as handle:
        pickle.dump(study.best_params, handle)

def get_args():
    parser = argparse.ArgumentParser(description="cVAE Trainer")
    parser.add_argument("model_savename", type=str, default="test",
                        help="Model save name (default=test)")
    parser.add_argument("study_savename", type=str, default="test",
                        help="Study save name (default=test)")
    parser.add_argument("trials", type=int, default=1000,
                        help="Number of study trials (default=1000)")
    parser.add_argument("--path", type=str, default="cvae/saves",
                        help="Parent folder path (default=cvae/saves)")
    parser.add_argument("--data-path", type=str, default="data/simulated_events.pickle",
                        help="Full path to pickle containing the training data (default=data/simulated_events.pickle)")
    parser.add_argument("--trial-epochs", type=int, default=200,
                        help="Number of epochs for each trial (default=200)")
    parser.add_argument("--disable-cuda", type=bool, default=True,
                        help="Whether to disable CUDA (default=True)")
    parser.add_argument("--gpu", type=str, default="",
                        help="GPU")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for torch (default=1)")
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    return args

def main():
    args = get_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(lambda trial : objective(args, trial), n_trials=args.trials, n_jobs=8, callbacks=[print_best])

    print(f"Best trial\n----------")
    best_trial = study.best_trial
    print(f"Value\n{best_trial.value}")

    save_best(args, study)

if __name__ == "__main__":
    main()
