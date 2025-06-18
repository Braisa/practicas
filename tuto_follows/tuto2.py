import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

df_housing = fetch_california_housing(as_frame=True).frame
df_housing = df_housing.apply(pd.to_numeric, errors="coerce", downcast="float")

features = df_housing.columns[:-1].drop("HouseAge")
prices = df_housing[df_housing.columns[-1]]

X = df_housing[features]

X_train, X_test, y_train, y_test = train_test_split(X, prices, test_size=0.2, random_state=3)

X_train_tensor = torch.from_numpy(X_train.values)
X_test_tensor = torch.from_numpy(X_test.values)
y_train_tensor = torch.from_numpy(y_train.values).unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test.values).unsqueeze(1)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

model = NN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train loss : {loss.item():.4f}, Test loss : {test_loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_train = model(X_train_tensor)

fig, ax = plt.subplots()

ax.plot(range(1, n_epochs+1), train_losses, color = "tab:blue", label = "Train loss")
ax.plot(range(1, n_epochs+1), test_losses, color = "tab:orange", label = "Test loss")

fig.savefig(f"tuto_follows/tuto2_losses.pdf", dpi = 300, bbox_inches = "tight")

error_test = mean_squared_error(y_test_tensor, y_pred_test)
error_train = mean_squared_error(y_train_tensor, y_pred_train)
print(f"Train error : {error_train:.4f}")
print(f"Test error : {error_test:.4f}")
