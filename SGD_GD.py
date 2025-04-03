import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# data
X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

# parameter
num_epochs = 300
lr = 0.1
num_hiddens = 3

class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.tanh(z1)
        z2 = self.linear2(a1)
        a2 = self.sigmoid(z2)
        return a2
    

# SGD
model_sgd = ShallowNN(2, num_hiddens)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr)
loss_fn = nn.BCELoss()
losses_sgd = []

for epoch in range(num_epochs):
    cost = 0.0
    for x, y in zip(X, Y):
        x_torch = torch.FloatTensor(x)
        y_torch = torch.FloatTensor([y])

        y_hat = model_sgd(x_torch)
        loss = loss_fn(y_hat, y_torch)

        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()

        cost += loss.item()
    cost /= len(X)
    losses_sgd.append(cost)


# GD
model_gd = ShallowNN(2, num_hiddens)
optimizer_gd = optim.SGD(model_gd.parameters(), lr=lr)
losses_gd = []


for epoch in range(num_epochs):
    optimizer_gd.zero_grad()
    cost = 0.0
    for x, y in zip(X, Y):
        x_torch = torch.FloatTensor(x)
        y_torch = torch.FloatTensor([y])
        y_hat = model_gd(x_torch)
        loss_val = loss_fn(y_hat, y_torch)
        cost += loss_val
    cost = cost / len(X)
    cost.backward()
    optimizer_gd.step()
    losses_gd.append(cost.item())


print("==SGD==")
for x, y in zip(X, Y):
    x_torch = torch.FloatTensor(x)
    y_hat = model_sgd(x_torch)
    print(f"Input: {x}, True: {y}, Predicted: {y_hat.item():.4f}")

print("==GD==")
for x, y in zip(X, Y):
    x_torch = torch.FloatTensor(x)
    y_hat = model_gd(x_torch)
    print(f"Input: {x}, True: {y}, Predicted: {y_hat.item():.4f}")

# 시각화
plt.plot(losses_sgd, label='SGD')
plt.plot(losses_gd, label='GD')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Comparison: SGD vs GD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()