import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from utils import scaler
import pandas as pd # type: ignore

class CounterDataset(Dataset):
    def __init__(self, counts, Ls, ps, xs, ys):
        self.counts = counts
        self.Ls = Ls
        self.ps = ps
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        sample = {
            "counts" : self.counts[index],
            "labels" : torch.tensor((
                self.Ls[index],
                self.ps[index],
                self.xs[index],
                self.ys[index]
            ), dtype=torch.float)
        }
        return sample

def create_counter_dataset(counts: torch.tensor, Ls: torch.tensor, ps: torch.tensor, xs: torch.tensor, ys: torch.tensor) -> CounterDataset:
    """
    Creates a pytorch dataset for the avalanche counter.

    Parameters
    ----------
    counts : torch.tensor
        Tensor that includes a list of the photon counts on each of the 132 detectors for each event.
    Ls : torch.tensor
        Tensor that includes a list of the detector collimating length for each event.
    ps : torch.tensor
        Tensor that includes a list of the scintillating gas mixture pressure for each event.
    xs : torch.tensor
        Tensor that includes a list of the x position of the ionizing particle for each event.
    ys : torch.tensor
        Tensor that includes a list of the y position of the ionizing particle for each event.
    
    Returns
    -------
    counter_dataset : CounterDataset
        Pytorch custom dataset for the avalanche counter.
    """
    counter_dataset = CounterDataset(counts, Ls, ps, xs, ys)
    return counter_dataset

def create_counter_scaler(data_names=None, data_min_values=None, data_max_values=None) -> scaler.Scaler:
    """
    Creates a scaler for the avalanche counter.
    
    Parameters
    ----------
    data_names : {"None", "other"}, optional
        Names that the scaler will assign to each of the variables. If None, default names will be used.
    data_min_values : {"None", "other"}, optional
        Minimum values for each of the variables. If None, default values will be used.
    data_max_values : {"None", "other"}, optional
        Maximum values for each of the variables. If None, default values will be used.
    
    Returns
    -------
    counter_scaler : scaler.Scaler
        Scaler for the avalanche counter.
    """
    if not data_names:
        data_names = ("counts", "L", "p", "x", "y")
    if not data_min_values:
        data_min_values = (0., 5., 10., -4000., -4000.)
    if not data_max_values:
        data_max_values = (905., 50., 50., 4000., 4000.)
    counter_scaler = scaler.create_scaler(data_names, data_min_values, data_max_values)
    return counter_scaler
