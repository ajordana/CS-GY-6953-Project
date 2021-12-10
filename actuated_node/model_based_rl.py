import argparse
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchdiffeq import odeint
from utils import (
    update_gamma,
    optimize_model,
    ReplayMemory,
    sample_traj,
    plot_pred,
    cost_function,
    evaluate_policy,
)
from pendulum_env import Pendulum


# Fix seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


# Choose policy
policy_A = False

if policy_A:
    ngamma = 2

    def create_policy(gamma):
        k = gamma[:2]
        tar = np.array([np.pi, 0.0])

        def policy(x):
            return np.vdot(k, x - tar)

        return policy

else:
    ngamma = 3

    def create_policy(gamma):
        k = gamma[:2]
        tar = gamma[2]

        def policy(x):
            return np.vdot(k, x) + tar

        return policy


nx = 2
nu = 1
n = nx + ngamma

std_gamma = 1
velocity_norm = 5.0


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, nx),
        )

    def forward(self, t, y):
        gam = y[:, nx:]
        z = y.new_zeros(gam.shape)
        th = y[:, :1]
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)
        th_dot = y[:, 1:2] / velocity_norm
        x = torch.cat((cos_th, sin_th, th_dot, gam), dim=1)
        state = self.net(x)
        return torch.cat((state, z), dim=1)


# Initialize model
func = ODEFunc()

# Load environment
env = Pendulum()
T = 10
time = torch.tensor(np.linspace(0.0, 1.0, T), dtype=torch.float32)


N_episodes = 200  # Number of episodes per learning iteration
memory = ReplayMemory(N_episodes)
BATCH_SIZE = 20

model_loss_list = []
cost_belief_list = []
cost_real_list = []

gamma_mean = np.zeros(ngamma)
print("\n")
N_it = 20
for ep in range(1, N_it + 1):
    print("iteration " + str(ep))
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    gamma_mean_norm = gamma_mean

    # Evaluate current policy
    policy = create_policy(gamma_mean)
    real_cost = evaluate_policy(np.zeros(2), env, policy, T)
    cost_real_list.append(real_cost)
    print("Real cost " + str(real_cost))

    # Collect data
    for i in range(N_episodes):
        gamma = gamma_mean + std_gamma * (2 * np.random.rand(ngamma) - 1)
        policy = create_policy(gamma)
        x0 = 2 * (2 * np.random.rand(2) - 1)
        traj = sample_traj(x0, env, gamma - gamma_mean_norm, policy, T, nx, ngamma)
        memory.push(traj)

    # Update model
    for i in range(5):
        loss = optimize_model(memory, func, time, optimizer, BATCH_SIZE)
        model_loss_list.append(loss)
    print("\n")

    # Update policy
    x0 = np.zeros(2)
    for i in range(5):
        gamma_mean, belief_cost_value = update_gamma(
            gamma_mean, gamma_mean_norm, func, time, x0, std_gamma, create_policy
        )
        print(
            "Belief cost " + str(belief_cost_value) + "    gamma =  " + str(gamma_mean)
        )
        cost_belief_list.append(belief_cost_value)
    print("\n")


# Plot a trajectory
T = 10
time = torch.tensor(np.linspace(0.0, 1.0, T), dtype=torch.float32)
policy = create_policy(gamma)
x0 = np.zeros(2)
traj = sample_traj(x0, env, gamma - gamma_mean_norm, policy, T, nx, ngamma)
plt.figure()
plot_pred(traj, func, time, nx)

# Plot cost and model loss according to learning iterations
plt.figure()
plt.plot(
    np.linspace(1, N_it, len(model_loss_list)),
    np.array(model_loss_list),
    label="Model loss",
)
plt.xlabel("Learning iterations")
plt.grid()
plt.legend()

plt.figure()
plt.plot(
    np.linspace(1, N_it, len(cost_belief_list)),
    np.array(cost_belief_list),
    label="Belief cost",
)
plt.plot(
    np.linspace(1, N_it, len(cost_real_list)),
    np.array(cost_real_list),
    label="Real cost",
)
plt.xlabel("Learning iterations")
plt.axis([1, N_it, 0, 10])
plt.legend()
plt.grid()
plt.show()
