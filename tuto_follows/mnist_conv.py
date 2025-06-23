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
            #nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, stride=1), # 26*26 -> 24*24
            #nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 24*24 -> 12*12
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1440, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

def train(args, model, loader, optimizer, epoch, no_print=False):
    initial_time = time()
    model.train()
    train_losses = []
    for batch_index, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if not no_print and batch_index % args.logging_interval == 0:
            print(f"Training epoch {epoch} [{batch_index*len(data)}/{len(loader.dataset)}]\tLoss: {loss.item():.6f}", end="\r")
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Training epoch {epoch} complete ({time()-initial_time:.2f}s elapsed)")
    return train_losses

def test(args, model, loader, no_print=False):
    initial_time = time()
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            output = model(data)
            losses.append(nn.functional.cross_entropy(output, target, reduction="mean").item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if not no_print:
                print(f"Testing progress [{batch_index*len(data)}/{len(loader.dataset)}] {100.*batch_index*len(data)/len(loader.dataset):.2f}%", end="\r")
    test_loss = np.mean(losses)
    accuracy = correct/len(loader.dataset)
    if not no_print:
        print(f"\033[K", end="\r")
        print(f"Testing, Average loss: {test_loss:.6f}\tAccuracy [{correct}/{len(loader.dataset)}] {100.*correct/len(loader.dataset):.2f}% ({time()-initial_time:.2f}s elapsed)")
    return test_loss, accuracy

def paint_losses(args, train_losses, test_losses, train_loader, test_loader):
    fig, ax = plt.subplots()

    train_spots = np.arange(1, 1+args.epochs*(1+(len(train_loader.dataset)//args.train_batch_size))) * args.train_batch_size
    test_spots = np.arange(1, 1+len(test_losses)) * len(train_loader.dataset)

    ax.plot(train_spots, train_losses, ls="solid", color="tab:blue", label="Train loss")
    ax.plot(test_spots, test_losses, ls="None", marker="o", color="tab:orange", label="Test loss")

    ax.xaxis.set_major_locator(plt.MultipleLocator(len(train_loader.dataset)))
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.legend(loc="best")
    
    fig.savefig(f"tuto_follows/{args.folder_name}_figs/{args.save_name}_losses.pdf", dpi=300, bbox_inches="tight")

def paint_accuracies(args, train_accuracies, test_accuracies):
    fig, ax = plt.subplots()

    spots = np.arange(1, 1+args.epochs)

    ax.plot(spots, train_accuracies, ls="dashed", marker="o", color="tab:blue", label="Train accuracy")
    ax.plot(spots, test_accuracies, ls="dashed", marker="o", color="tab:orange", label="Test accuracy")

    ax.set_ylim(top=1)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(.005))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(.001))

    ax.legend(loc="best")

    fig.savefig(f"tuto_follows/{args.folder_name}_figs/{args.save_name}_accuracies.pdf", dpi=300, bbox_inches="tight")

def main():
    main_time = time()
    
    parser = argparse.ArgumentParser(description="CNN MNIST")
    
    # File settings
    parser.add_argument("--folder-name", type=str, default="mnist_cnn",
                        help="input folder for storing figures (default=mnist_conv)")
    parser.add_argument("--save-name", type=str, default="mnist_cnn",
                        help="input file save name (default=mnist_conv)")

    # Training settings
    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="input training batch size (default=64)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="input testing batch size (default=1000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="input number of epochs (default=3)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="input learning rate (default=0.01)")
    parser.add_argument("--random-seed", type=int, default=1,
                        help="input random seed (default=1)")
    parser.add_argument("--logging-interval", type=int, default=50,
                        help="input logging interval (default=50)")
    parser.add_argument("--optim", type=str, default="Adam",
                        help="input optimizer (default=Adam)")
    parser.add_argument("--seed", type=int, default=1,
                        help="input random seed (default=1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

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
    match args.optim:
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(1, args.epochs+1):
        train_log = train(args, model, train_loader, optimizer, epoch)
        _, train_accuracy = test(args, model, train_loader, no_print=False)
        test_loss, test_accuracy = test(args, model, test_loader)

        train_losses.append(train_log)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    paint_losses(args, np.ravel(train_losses), test_losses, train_loader, test_loader)
    paint_accuracies(args, train_accuracies, test_accuracies)

    main_total = time()-main_time
    main_days = int(main_total // 86400)
    main_hours = int((main_total % 86400) // 3600)
    main_minutes = int(((main_total % 86400) % 3600) // 60)
    main_seconds = int(((main_total % 86400) % 3600) % 60)
    print(f"Program complete ({main_days:02}:{main_hours:02}:{main_minutes:02}:{main_seconds:02} elapsed)")

if __name__ == "__main__":
    main()
