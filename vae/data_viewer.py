import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle("vae/simulated_events.pickle")

dcols = np.array(list(df["dcol"].values()))
ps = np.array(list(df["p"].values()))
xs = np.array(list(df["x"].values()))
ys = np.array(list(df["y"].values()))
ns = np.array(list(df["nphotons"].values()))

def ax_setup(ax, title):
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Photon count")

    ax.set_xlim(left=1, right=33*4)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(plt.MultipleLocator(base=33))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    ax.legend(loc="best")
    ax.set_title(title)

# Comparing for different momentum, with centered particle and same collimation

p_indexes = (
    np.where((dcols==5.) & (ps==10.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==20.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==40.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==50.) & (xs==0.) & (ys==0.))[0][0],
)

# Comparing for different position, with same momentum and collimation

pos_indexes = (
    np.where((dcols==5.) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==1000.) & (ys==1000.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==2000.) & (ys==2000.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==3000.) & (ys==3000.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==4000.) & (ys==4000.))[0][0],
)

# Comparing for off-centering in one direction, with same momentum and collimation

offcentering_indexes = (
    np.where((dcols==5.) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==1000.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==-1000.) & (ys==0.))[0][0],
    np.where((dcols==5.) & (ps==30.) & (xs==-2000.) & (ys==0.))[0][0],
)

# Comparing for different collimation lengths, for centered particles with same momentum

collimation_indexes = (
    np.where((dcols==5.) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==16.25) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==27.5) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==38.75) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
    np.where((dcols==50.) & (ps==30.) & (xs==0.) & (ys==0.))[0][0],
)

# Comparing for different collimation lengths, for offset particles with same momentum

offsetcollimation_indexes = (
    np.where((dcols==5.) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
    np.where((dcols==16.25) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
    np.where((dcols==27.5) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
    np.where((dcols==38.75) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
    np.where((dcols==50.) & (ps==30.) & (xs==2000.) & (ys==0.))[0][0],
)


all_indexes = (
    p_indexes,
    pos_indexes,
    offcentering_indexes,
    collimation_indexes,
    offsetcollimation_indexes
)

fig_names = (
    "momentum_comparison",
    "position_comparison",
    "offcentering_comparison",
    "collimation_comparison",
    "offset_collimation_comparison"
)

fig_titles = (
    "Centered particles, dcol=5",
    "p=30, dcol=5",
    "p=30, dcol=5, y=0",
    "Centered particles, p=30",
    "p=30, off_x=2000, y=0"
)

label_names = (
    "p",
    "pos",
    "off_x",
    "dcol",
    "dcol"
)

label_values = (
    ("10", "20", "30", "40", "50"),
    ("(0,0)", "(1000,1000)", "(2000,2000)", "(3000,3000)", "(4000,4000)"),
    ("0", "1000", "2000", "-1000", "-2000"),
    ("5", "16.25", "27.5", "38.75", "50"),
    ("5", "16.25", "27.5", "38.75", "50")
)

spots = np.arange(1, 1+33*4)

for indexes, name, title, lab, values in zip(all_indexes, fig_names, fig_titles, label_names, label_values):

    fig, ax = plt.subplots()

    for i, (index, value) in enumerate(zip(indexes, values)):
        ax.plot(spots, ns[index], label=f"{lab}={value}")
    
    ax_setup(ax, title)
    fig.savefig(f"vae/vae_figs/{name}.pdf", dpi=300, bbox_inches="tight")
