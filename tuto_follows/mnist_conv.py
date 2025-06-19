import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms, datasets #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt
from utils.early_stopper import EarlyStopper
from time import time

folder_name = "mnist_conv"
save_name = "mnist_conv"

n_filters = 10

n_epochs = 5
batch_size_train, batch_size_test = 512, 512
learning_rate = 1e-2

torch.manual_seed(1)

loader_train = DataLoader(
    datasets.MNIST(root="./mnist_data", train=True, download=False, transform=transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True
)

loader_test = DataLoader(
    datasets.MNIST(root="./mnist_data", train=False, download=False, transform=transforms.ToTensor()),
    batch_size=batch_size_test, shuffle=True
)

class Conv_NN(nn.Module):
    def __init__(self, n_filters=10):
        super(Conv_NN, self).__init__()
        self.n_filters = n_filters
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.n_filters, kernel_size=3), #28*28 -> 26*26
            nn.BatchNorm2d(self.n_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1) #26*26 -> 24*24
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(24*24*self.n_filters, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
            nn.ReLU(),
            nn.Linear(36, 10),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, 24*24*self.n_filters) # flattens
        x = self.linear_layers(x)
        return x

model = Conv_NN(n_filters=n_filters)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

early_stopper = EarlyStopper(patience=20, loss_delta=0.001)

train_losses, test_losses = [], []

def train(epoch):
    initial_time = time()
    model.train()
    loss_sum = 0
    correct = 0
    for data, target in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += target.eq(predictions).sum()
    train_losses.append(loss_sum/len(loader_train.dataset))
    epoch_lapse = time() - initial_time
    print(f"Epoch {epoch+1}/{n_epochs}, Train loss : {train_losses[-1]:.4f}, Train lapse : {epoch_lapse:.2f}, Train accuracy : {correct/len(loader_train.dataset):.3f}")

def test(epoch):
    initial_time = time()
    model.eval()
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader_test:
            outputs = model(data)
            loss_sum += criterion(outputs, target)
            predictions = torch.argmax(outputs, dim=1)
            correct += target.eq(predictions).sum()
        test_losses.append(loss_sum/len(loader_test.dataset))
    epoch_lapse = time() - initial_time
    print(f"Epoch {epoch+1}/{n_epochs}, Test loss : {test_losses[-1]:.4f}, Test lapse : {epoch_lapse:.2f}, Test accuracy : {correct/len(loader_test.dataset):.3f}")

for epoch in range(n_epochs):

    train(epoch)
    test(epoch)
    
    if early_stopper.check_early_stop(test_losses[-1]):
        break

fig, ax = plt.subplots()

ax.plot(range(1, epoch+2), train_losses, color = "tab:blue", label = "Train loss")
ax.plot(range(1, epoch+2), test_losses, color = "tab:orange", label = "Test loss")

ax.legend(loc = "best")
ax.set_title("conv:24*24*10:128:32:10")

fig.savefig(f"tuto_follows/{folder_name}_figs/{save_name}_losses.pdf", dpi = 300, bbox_inches = "tight")
