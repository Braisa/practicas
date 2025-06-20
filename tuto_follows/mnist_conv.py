import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms, datasets #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt
from utils.early_stopper import EarlyStopper
from time import time
import numpy as np
import argparse

class Conv_NN(nn.Module):
    def __init__(self):
        super(Conv_NN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, stride=1), # 28*28 -> 26*26
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, stride=1), # 26*26 -> 24*24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 24*24 -> 12*12
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1440, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

def train(args, model, loader, optimizer, epoch):
    model.train()
    logged_losses = []
    for batch_index, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % args.logging_interval == 0:
            print(f"Training epoch {epoch} [{batch_index*len(data)}/{len(loader.dataset)}]\tLoss: {loss.item():.6f}")
            logged_losses.append(loss.item())
    return logged_losses

def test(args, model, loader):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            losses.append(nn.functional.nll_loss(output, target, reduction="mean").item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = np.mean(losses)
    print(f"Testing, Average loss: {test_loss:.6f}, Accuracy {100.*correct/len(loader.dataset):.2f}%")
    return test_loss

def main():
    # File settings
    folder_name = "mnist_conv"
    save_name = "mnist_conv_example"

    # Training settings
    parser = argparse.ArgumentParser(description="CNN MNIST")

    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="input training batch size (default=64)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="input testing batch size (default=1000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="input number of epochs (default=3)")
    parser.add_argument("--learning-rate", type=float, default=1e-2,
                        help="input learning rate (default=0.01)")
    parser.add_argument("--random-seed", type=int, default=1,
                        help="input random seed (default=1)")
    parser.add_argument("--logging-interval", type=int, default=50,
                        help="input logging interval (default=50)")
    args = parser.parse_args()

    torch.manual_seed(1)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.1307,),(.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST(root="./mnist_data", train=True, download=False, transform=trans),
        batch_size=args.train_batch_size, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root="./mnist_data", train=False, download=False, transform=trans),
        batch_size=args.test_batch_size, shuffle=True
    )

    model = Conv_NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs+1):
        train_loss = train(args, model, train_loader, optimizer, epoch)
        test_loss = test(args, model, test_loader)
    
    """
    fig, ax = plt.subplots()

    ax.plot(range(1, epoch+2), train_losses, color = "tab:blue", label = "Train loss")
    ax.plot(range(1, epoch+2), test_losses, color = "tab:orange", label = "Test loss")

    ax.legend(loc = "best")
    ax.set_title("conv10:conv20:320:100:10")

    fig.savefig(f"tuto_follows/{folder_name}_figs/{save_name}_losses.pdf", dpi = 300, bbox_inches = "tight")
    """

if __name__ == "__main__":
    main()
