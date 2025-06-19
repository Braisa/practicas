import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms, datasets #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt
from utils.early_stopper import EarlyStopper

folder_name = "tuto2_mnist"
save_name = "tuto2_mnist_batch_512_veryslow"

mnist = datasets.MNIST(root="./mnist_data", train=True, download=False, transform=transforms.ToTensor())

X_data = mnist.data.numpy().reshape(-1, 28*28) / 255.0
y_data = mnist.targets.numpy()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=3)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

X_train_tensor_norm = (X_train_tensor - X_train_tensor.mean()) / X_train_tensor.std()
X_test_tensor_norm = (X_test_tensor - X_test_tensor.mean()) / X_test_tensor.std()

class BuildDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

dataset_train = BuildDataset(X_train_tensor_norm, y_train_tensor)
dataset_test = BuildDataset(X_test_tensor_norm, y_test_tensor)

loader_train = DataLoader(dataset=dataset_train, batch_size=512)
loader_test = DataLoader(dataset=dataset_test, batch_size=512)

model = NN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

early_stopper = EarlyStopper(patience=20, loss_delta=0.001)

n_epochs = 100
train_losses, test_losses = [], []

for epoch in range(n_epochs):

    model.train()
    for X, y in loader_train:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        for X, y in loader_test:
            test_outputs = model(X)
            test_loss = criterion(test_outputs, y)
        test_losses.append(test_loss.item())
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train loss : {loss.item():.4f}, Test loss : {test_loss.item():.4f}")

    if early_stopper.check_early_stop(test_loss):
        break

model.eval()
with torch.no_grad():
    y_pred_test = torch.argmax(model(X_test_tensor_norm), dim=1)
    y_pred_train = torch.argmax(model(X_train_tensor_norm), dim=1)

fig, ax = plt.subplots()

ax.plot(range(1, epoch+2), train_losses, color = "tab:blue", label = "Train loss")
ax.plot(range(1, epoch+2), test_losses, color = "tab:orange", label = "Test loss")

ax.legend(loc = "best")
ax.set_title("28*28:128:32:10")

fig.savefig(f"tuto_follows/{folder_name}_figs/{save_name}_losses.pdf", dpi = 300, bbox_inches = "tight")

acc_test = accuracy_score(y_test_tensor, y_pred_test)
acc_train = accuracy_score(y_train_tensor, y_pred_train)

print(f"Train accuracy : {acc_train:.3f}\nTest accuracy : {acc_test:.3f}")
