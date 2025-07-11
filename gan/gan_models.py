import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, inner_dim, hidden_dim, output_dim, hidden_layers=2):
        super(Generator, self).__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        net_layers = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid()
        )
        for _ in range(hidden_layers-1):
            net_layers.append(nn.Linear(hidden_dim, hidden_dim))
            net_layers.append(nn.Sigmoid())
        self.net_layers = net_layers
    
    def forward(self, z, l):
        total_input = torch.cat((z,l), dim=1)
        x = self.net_layers(total_input)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, big_hidden_dim, small_hidden_dim, big_hidden_layers=2, small_hidden_layers=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.big_hidden_dim = big_hidden_dim
        self.big_hidden_layers = big_hidden_layers
        self.small_hidden_dim = small_hidden_dim
        self.small_hidden_layers = small_hidden_layers

        net_layers = nn.Sequential(
            nn.Linear(input_dim, big_hidden_dim),
            nn.BatchNorm1d(big_hidden_dim),
            nn.Sigmoid()
        )
        for _ in range(big_hidden_layers-1):
            net_layers.append(nn.Linear(big_hidden_dim, big_hidden_dim))
            net_layers.append(nn.Sigmoid())
        net_layers.append(nn.Linear(big_hidden_dim, small_hidden_dim))
        net_layers.append(nn.Sigmoid())
        for _ in range(small_hidden_layers-1):
            net_layers.append(nn.Linear(small_hidden_dim, small_hidden_dim))
            net_layers.append(nn.Sigmoid())
        net_layers.append(nn.Linear(small_hidden_dim, 1))
        net_layers.append(nn.Sigmoid())
        self.net_layers = net_layers
    
    def forward(self, x, l):
        total_input = torch.cat((x,l), dim=1)
        val = self.net_layers(total_input)
        return val

def create_GAN(nonlabel_input_dim: int, label_input_dim: int, gen_hidden_dim: int, big_hidden_dim: int, small_hidden_dim: int, latent_dim: int, gen_hidden_layers: int, big_hidden_layers: int, small_hidden_layers: int):
    """
    Creates a Generator and Discriminator that compose a system of Generative Adversarial Networks.

    Parameters
    ----------
    nonlabel_input_dim : int
        Size of the non-label input that the Generator will output, and the Discriminator will receive.
    label_input_dim : int
        Size of the label input that the Generator will output, and the Discriminator will receive.
    gen_hidden_dim : int
        Size of the hidden layers of the Generator's neural network.
    big_hidden_dim : int
        Size of the big hidden layers of the Discriminator's neural network.
    small_hidden_dim : int
        Size of the small hidden layers of the Discriminator's neural network.
    latent_dim : int
        Size of the latent neurons in the Generator's neural network. Note that this excludes the amount of labels.
    gen_hidden_layers : int
        Amount of hidden layers in the Generator's neural network.
    big_hidden_layers : int
        Amount of big hidden layers in the Discriminator's neural network.
    small_hidden_layers : int
        Amount of small hidden layers in the Discriminator's neural network.
    
    Returns
    -------
    generator : Generator
        Generator model created.
    discriminator : Discriminator
        Discriminator model created.
    """

    generator = Generator(
        inner_dim=latent_dim,
        hidden_dim=gen_hidden_dim,
        output_dim=nonlabel_input_dim+label_input_dim,
        hidden_layers=gen_hidden_layers
    )
    discriminator = Discriminator(
        input_dim=nonlabel_input_dim+label_input_dim,
        big_hidden_dim=big_hidden_dim,
        small_hidden_dim=small_hidden_dim,
        big_hidden_layers=big_hidden_layers,
        small_hidden_layers=small_hidden_layers
    )
    return generator, discriminator
