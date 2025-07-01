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

from labelled_vae import cVAE, DetectorDataset, Scaler, create_scaler

def get_args():
    parser = argparse.ArgumentParser(description="cVAE Figure Printer")
    
    # Mandatory arguments
    parser.add_argument("savename", type=str,
                        help="input save name for printed figure")
    
    # Figure arguments
    parser.add_argument("--dpi", type=int, default=300,
                        help="input saved figure dpi (default=300)")
    parser.add_argument("--bbox-inches", type=str, default="tight",
                        help="input boundary box inches (default=tight)")

    # Storage arguments
    parser.add_argument("--folder-name", type=str, default="vae/cvae",
                        help="input parent folder directory (default=vae/cvae)")
    parser.add_argument("--subdirectory", type=str, default="bulk",
                        help="input subdirectory folder where figures will be stored (default=bulk)")

    return parser.parse_args()

def load_data(args, scaler):
    df = pd.DataFrame(pd.read_pickle("vae/simulated_events.pickle"))
    
    counts = scaler.downscale("counts", torch.tensor(list(df["nphotons"].values), dtype=torch.float))
    Ls     = scaler.downscale("L",      torch.tensor(list(df["dcol"].values), dtype=torch.float))
    ps     = scaler.downscale("p",      torch.tensor(list(df["p"].values), dtype=torch.float))
    xs     = scaler.downscale("x",      torch.tensor(list(df["x"].values), dtype=torch.float))
    ys     = scaler.downscale("y",      torch.tensor(list(df["y"].values), dtype=torch.float))

    return DetectorDataset(counts, Ls, ps, xs, ys)

def main():
    
    args = get_args()
    scaler = create_scaler()
    data = load_data(args, scaler)


if __name__ == "__main__":
    main()
