import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, inner_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.encoding_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.mean_layer = nn.Linear(hidden_dim, inner_dim)
        self.logvar_layer = nn.Linear(hidden_dim, inner_dim)
    
    def forward(self, x, l):
        total_input = torch.cat((x,l), dim=1)
        x = self.encoding_layers(total_input)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, inner_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.decoding_layers = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
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
    def __init__(self, Encoder, Decoder, latent_dim=3, label_dim=4):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
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

def create_cVAE(nonlabel_input_dim: int, label_input_dim: int, hidden_dim: int, latent_dim: int) -> cVAE:
    """
    Creates a Conditional Variational Autoencoder model.

    Parameters
    ----------
    nonlabel_input_dim : int
        Size of the non-label input that the encoder will receive.
    label_input_dim : int
        Size of the label input that the encoder will receive.
    hidden_dim : int
        Size of the hidden layers of the neural network.
    latent_dim : int
        Size of the latent neurons in the neural network. Note that this excludes the amount of labels.
    
    Returns
    -------
    cvae : cVAE
        Conditional Variational Autoencoder model created.
    """
    enc = Encoder(input_dim=nonlabel_input_dim+label_input_dim, hidden_dim=hidden_dim, inner_dim=latent_dim)
    dec = Decoder(inner_dim=latent_dim+label_input_dim, hidden_dim=hidden_dim, output_dim=nonlabel_input_dim+label_input_dim)
    cvae = cVAE(Encoder=enc, Decoder=dec, latent_dim=latent_dim, label_dim=label_input_dim)
    return cvae
