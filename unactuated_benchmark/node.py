import argparse
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import makedirs
from utils import load_dataset
import torch.nn.functional as F
from torchdiffeq import odeint


makedirs("figures")
makedirs("png")
makedirs("log")

# Choose experiment: Pendulum or Cartpole
name = "Cartpole"


n = 2
if name == "Cartpole":
    n = 4

# Load data
t_train = np.load("datasets/" + name + "_time_train.npy")
t_train = torch.tensor(t_train)
t_test = np.load("datasets/" + name + "_time_test.npy")
t_test = torch.tensor(t_test)

# ODE model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n, 512),
            nn.ReLU(),
            nn.Linear(512, n),
        )

    def forward(self, t, y):
        return self.net(y)


def train(N_epochs, batch_size=50, plot=False):
    # Initialize model and optimizer
    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    # Load data
    train_loader, test_loader, mean, std = load_dataset(name, batch_size)

    loss_list = []
    avg_loss = 0.0
    for Epoch in range(N_epochs):
        # Train loop
        for batch in train_loader:
            batch_y0 = batch[:, 0]
            batch_y = np.transpose(batch, (1, 0, 2))
            optimizer.zero_grad()
            pred_y = odeint(func, batch_y0, t_train)
            loss = F.mse_loss(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().numpy()
        avg_loss = avg_loss / len(train_loader)

        # Test loop
        with torch.no_grad():
            loss = 0
            for batch in test_loader:
                batch_y0 = batch[:, 0]
                batch_y = np.transpose(batch, (1, 0, 2))
                pred_y = odeint(func, batch_y0, t_test)
                loss += F.mse_loss(pred_y, batch_y).detach().cpu().numpy()
            loss = loss / len(test_loader)
            pred_y = pred_y[:, :1]
            true_y = batch_y[:, :1]
            print("Epoch {:02d} | Total Loss {:.6f}".format(Epoch, loss))
        loss_list.append(loss)

    # Plot a testing trajectory and its prediction
    if plot:
        pred_y = pred_y[:, 0].detach().cpu().numpy() * std + mean
        true_y = batch_y[:, 0].detach().cpu().numpy() * std + mean

        if n == 2:
            plt.plot(t_test, true_y[:, 0], "--", label="theta GT")
            plt.plot(t_test, pred_y[:, 0], label="theta hat")
            plt.plot(t_test, true_y[:, 1], "--", label="theta dot GT")
            plt.plot(t_test, pred_y[:, 1], label="theta dot hat")
        else:
            plt.plot(t_test, true_y[:, 0], "--", label="x GT")
            plt.plot(t_test, pred_y[:, 0], label="x hat")
            plt.plot(t_test, true_y[:, 1], "--", label="x dot GT")
            plt.plot(t_test, pred_y[:, 1], label="x dot hat")

            plt.plot(t_test, true_y[:, 2], "--", label="theta GT")
            plt.plot(t_test, pred_y[:, 2], label="theta hat")
            plt.plot(t_test, true_y[:, 3], "--", label="theta dot GT")
            plt.plot(t_test, pred_y[:, 3], label="theta dot hat")

        plt.legend()
        plt.savefig("figures/" + name + "node")
    return np.array(loss_list)


# Train for 100 epochs with 10 different seeds
N = 10
N_epochs = 100
loss = np.zeros((N, N_epochs))
for i in range(N):
    loss[i] = train(N_epochs, plot=True)
np.save("log/" + name + "node_loss", loss)
