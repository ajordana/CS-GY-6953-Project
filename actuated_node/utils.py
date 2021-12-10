import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import random
import matplotlib.pyplot as plt

# Fix seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def sample_traj(x0, env, gamma, policy, T, nx, ngamma):
    n = nx + ngamma
    traj = np.zeros((T, n))
    state = x0
    traj[0, :nx] = state
    for i in range(T - 1):
        state = env.step(state, policy(state))
        traj[i + 1, :nx] = state
    traj[:, nx:] = gamma
    return traj


def plot_pred(traj, func, time, nx):
    with torch.no_grad():
        batch = torch.tensor(traj, dtype=torch.float32).unsqueeze(0)
        batch_y0 = batch[:, 0]
        pred_y = odeint(func, batch_y0, time).squeeze().detach().numpy()

    traj[:, :nx] = traj[:, :nx]
    pred_y = pred_y[:, :nx]
    time = time.detach().numpy()
    plt.plot(time, traj[:, 0], "--", label="GT")
    plt.plot(time, pred_y[:, 0], label="Belief")
    plt.plot(time[-1], np.pi, "o")
    plt.legend()


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

    def push(self, traj):
        """Save a transition"""
        self.memory.append(traj)
        if self.__len__() > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return np.array(random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


def optimize_model(memory, func, time, optimizer, BATCH_SIZE, n_update=30):
    if len(memory) < BATCH_SIZE:
        return

    for i in range(n_update):
        batch = memory.sample(BATCH_SIZE)
        batch = torch.tensor(batch, dtype=torch.float32)

        batch_y0 = batch[:, 0]
        batch_y = np.transpose(batch, (1, 0, 2))
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0, time)
        loss = F.mse_loss(pred_y, batch_y)
        loss.backward()
        optimizer.step()
    print("model loss = " + str(loss.item()))
    return loss.item()


def cost_function(traj, u):
    lx = np.mean((traj[:, 0] - np.pi) ** 2) + 1e-2 * np.mean(traj[:, 1] ** 2)
    lu = 1e-3 * np.mean(u ** 2)
    return lx + lu


def evaluate_policy(x0, env, policy, T):
    traj = np.zeros((T + 1, len(x0)))
    u = np.zeros((T, len(x0)))
    state = x0
    traj[0] = state
    for i in range(T - 1):
        tau = policy(state)
        state = env.step(state, tau)
        traj[i + 1] = state
        u[i] = tau
    return cost_function(traj, u)


def update_gamma(gamma_mean, gnorm, func, time, x0, std_gamma, create_policy, N=100):
    cost_list = []
    gamma_list = []
    x0 = time.new_tensor(x0, dtype=torch.float32)
    for it in range(N):
        # Sample a policy
        gamma = gamma_mean + std_gamma * (2 * np.random.rand(len(gamma_mean)) - 1)
        gamma_list.append(gamma)

        # Predict a trajectory given our model
        delta_gamma = time.new_tensor(gamma - gnorm, dtype=torch.float32)
        y0 = torch.cat((x0, delta_gamma)).unsqueeze(0)
        pred_y = odeint(func, y0, time).squeeze()[:, :2].detach().numpy()

        # Evaluate cost of the trajectory
        policy = create_policy(gamma)
        u = np.array([policy(pred_y[i]) for i in range(len(pred_y))])
        cost = cost_function(pred_y, u)
        cost_list.append(cost)

    # return the average of the best policies
    k = 10
    cost_list = np.array(cost_list)
    gamma_list = np.array(gamma_list)
    idx = np.argpartition(cost_list, k)
    best_gamma = gamma_list[idx[:k]]
    best_cost = cost_list[idx[:k]]
    return np.mean(best_gamma, axis=0), np.mean(best_cost)
