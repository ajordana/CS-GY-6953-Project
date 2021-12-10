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
from torchdiffeq import odeint_adjoint as odeint


makedirs("figures")
makedirs("png")
makedirs("log")


# Choose experiment: Pendulum or Cartpole
name = "Pendulum"


n = 2
if name == "Cartpole":
    n = 4

# Resnet model
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n, 512),
            nn.ReLU(),
            nn.Linear(512, n),
        )

    def forward(self, y):
        return self.net(y)

    def unroll(self, x0, N):
        pred = x0.new_zeros((x0.shape[0], N, x0.shape[1]))
        x = x0
        pred[:, 0] = x
        for i in range(N - 1):
            x = x + self.forward(x)
            pred[:, i + 1] = x
        return pred


def train(N_epochs, batch_size=50, plot=False):
    # Initialize model and optimizer
    func = resnet()
    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    # Load data
    train_loader, test_loader, mean, std = load_dataset(name, batch_size)

    loss_list = []
    for Epoch in range(N_epochs):
        # Train loop
        avg_loss = 0
        for batch in train_loader:
            batch_y0 = batch[:, 0]
            batch_y = batch
            N = batch_y.shape[1]
            optimizer.zero_grad()
            pred_y = func.unroll(batch_y0, N)
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
                batch_y = batch
                N = batch.shape[1]
                pred_y = func.unroll(batch_y0, N)
                loss += F.mse_loss(pred_y, batch_y).detach().cpu().numpy()
            loss = loss / len(test_loader)
            loss_list.append(loss)
    print("Epoch {:02d} | Test Loss {:.6f}".format(Epoch, loss))

    return np.array(loss_list)


# Train for 100 epochs with 10 different seeds
N = 10
N_epochs = 100
loss = np.zeros((N, N_epochs))
for i in range(N):
    loss[i] = train(N_epochs)
np.save("log/" + name + "resnet_loss", loss)
