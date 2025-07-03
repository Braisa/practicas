import matplotlib.pyplot as plt
import numpy as np
import torch # type: ignore
import torch.nn as nn # type:ignore
import pandas as pd # type: ignore
import pickle
import argparse
from pathlib import Path
from cvae import Encoder, Decoder, cVAE

def load_data(args):
    data = pd.read_pickle(f"{args.path}/{args.savename}/{args.savename}_data.pickle")
    return data

def load_model(args, data):
    model_args = data["args"]
    nlid, lid = model_args.nonlabel_input_dim, model_args.label_input_dim
    hd, ld = model_args.hidden_dim, model_args.latent_dim

    enc = Encoder(input_dim=nlid+lid, hidden_dim=hd, inner_dim=ld)
    dec = Decoder(inner_dim=lid+ld, hidden_dim=hd, output_dim=nlid+lid)
    model = cVAE(Encoder=enc, Decoder=dec, latent_dim=ld, label_dim=lid)

    model.load_state_dict(torch.load(f"{data["args"].path}/{data["args"].savename}/{data["args"].savename}_model.pt"))
    model.eval()
    return model

def generate_output(args, data, model, true_labels, return_true_counts=True):
    label_names = list(data["scaler"].scaling_data.keys())[1:]

    labels_up = true_labels
    labels_up_tensor = torch.tensor(labels_up, dtype=torch.float).unsqueeze(0)

    torch.manual_seed(data["args"].seed)
    model.eval()
    with torch.no_grad():
        zs = torch.randn(1, model.latent_dim)
        t = model.Decoder(zs, labels_up_tensor, dim=1)[0]
    gen_counts, gen_labels = t[:data["args"].nonlabel_input_dim], t[data["args"].nonlabel_input_dim:]
    upscaled_gen_counts = data["scaler"].upscale("counts", gen_counts)
    upscaled_gen_labels = []
    for gen_label, label_name in zip(gen_labels, label_names):
        upscaled_gen_labels.append(data["scaler"].upscale(label_name, gen_label))
    
    if not return_true_counts:
        return upscaled_gen_counts, upscaled_gen_labels
    else:
        L, p, x, y = labels_up
        L_truth = L == data["data"]["L"]
        p_truth = p == data["data"]["p"]
        x_truth = x == data["data"]["x"]
        y_truth = y == data["data"]["y"]
        index = np.where(L_truth & p_truth & x_truth & y_truth)[0][0]
        true_counts = data["data"]["counts"][index]
        upscaled_true_counts = data["scaler"].upscale("counts", true_counts)
        return upscaled_gen_counts, upscaled_gen_labels, upscaled_true_counts

def comparison_ax(args, data, ax, true_counts, true_labels, gen_counts, gen_labels, minor=True):
    label_names = list(data["scaler"].scaling_data.keys())[1:]
    true_title, gen_title = "True: ", "Gen: "
    for label_name, true_label, gen_label in zip(label_names, true_labels, gen_labels):
        true_title += f"{label_name}={true_label:.2f}"
        gen_title += f"{label_name}={gen_label:.2f}"
    
    spots = np.arange(1, 1+data["args"].nonlabel_input_dim)
    ax.step(spots, true_counts, color = "tab:blue")
    ax.step(spots, gen_counts, color = "tab:orange")

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=data["args"].nonlabel_input_dim/4))
    if minor:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Photon count")

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=1,right=data["args"].nonlabel_input_dim)

    ax.set_title(f"{true_title}\n{gen_title}")

def get_true_labels(args, data):
    L_in = np.unique(data["data"]["L"])[args.gen_input[0]]
    p_in = np.unique(data["data"]["p"])[args.gen_input[1]]
    x_in = np.unique(data["data"]["x"])[args.gen_input[2]]
    y_in = np.unique(data["data"]["y"])[args.gen_input[3]]
    true_labels = (L_in, p_in, x_in, y_in)
    L_in_up = data["scaler"].upscale("L", L_in)
    p_in_up = data["scaler"].upscale("p", p_in)
    x_in_up = data["scaler"].upscale("x", x_in)
    y_in_up = data["scaler"].upscale("y", y_in)
    true_labels_up = (L_in_up, p_in_up, x_in_up, y_in_up)
    return true_labels, true_labels_up

def comparison_fig(args, data, model):
    fig, ax = plt.subplots()
    
    true_labels, true_labels_up = get_true_labels(args, data)
    gen_counts_up, gen_labels_up, true_counts_up = generate_output(args, data, model, true_labels)

    comparison_ax(args, data, ax, true_counts_up, true_labels, gen_counts_up, gen_labels_up)

    full_path = f"{data["args"].path}/{data["args"].savename}/figs"
    Path(f"{full_path}").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{full_path}/{args.figname}_comp.pdf", dpi=300, bbox_inches="tight")

def get_args():
    parser = argparse.ArgumentParser("cVAE Comparison Figure Printer")
    # File settings
    parser.add_argument("figname", type=str, default="test",
                        help="Figure save name")
    # Generation settings
    parser.add_argument("gen_input", type=int, nargs="+",
                        help="Input (L,p,x,y) values for generated output, as indexes on the unique values lists")
    # Model settings
    parser.add_argument("--path", type=str, default="cvae/saves",
                        help="Model save path")
    parser.add_argument("--savename", type=str, default="test",
                        help="Model save name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data = load_data(args)
    model = load_model(args, data)
    comparison_fig(args, data, model)

if __name__ == "__main__":
    main()
