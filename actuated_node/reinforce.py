import matplotlib.pyplot as plt
import numpy as np
from utils import cost_function
from pendulum_env import Pendulum

# Choose policy
policy_A = True

# Define the policy and its log gradient
if policy_A:
    ngamma = 2

    def create_policy(gamma):
        k = gamma[:2]
        tar = np.array([np.pi, 0.0])

        def policy(x):
            return np.vdot(k, x - tar)

        return policy

    def grad_log_policy(state, action, gamma):
        tar = np.array([np.pi, 0.0])
        mu = np.vdot(gamma, state - tar)
        return (action - mu) * (state - tar)

else:
    ngamma = 3

    def create_policy(gamma):
        k = gamma[:2]
        tar = gamma[2]

        def policy(x):
            return np.vdot(k, x) + tar

        return policy

    def grad_log_policy(state, action, gamma):
        tar = np.array([np.pi, 0.0])
        mu = np.vdot(gamma[:2], state) + gamma[2]
        dmu = np.zeros(ngamma)
        dmu[:2] = state
        dmu[2] = 1.0
        return (action - mu) * dmu


# Compute the gradient over a trjectory
def grad_traj(traj, u, gamma):
    n = len(traj) - 1
    grad = np.zeros(ngamma)
    for i in range(n):
        grad += grad_log_policy(traj[i], u[i], gamma)
    return grad / n


# Sample a trajectory
def sample_random_traj(x0, policy):
    noise = np.random.normal(0, std, T)
    traj = np.zeros((T + 1, 2))
    u = np.zeros(T)
    state = x0
    traj[0] = state
    for i in range(T):
        tau = policy(state) + noise[i]
        state = env.step(state, tau)
        traj[i + 1] = state
        u[i] = tau
    return traj, u


lr = 1e-1  # learning rate
N_episodes = 1000000
std = 0.1


# Initialise environment
env = Pendulum()
T = 10


# Compute the expected return for the baseline
cost_mean = 0

# Initialize policy
gamma = np.zeros(ngamma)

cost_list = []
for it in range(N_episodes):
    # Sample a trajectory
    policy = create_policy(gamma)
    x0 = np.zeros(2)
    traj, u = sample_random_traj(x0, policy)

    # Evaluate cost
    cost = cost_function(traj, u)
    cost_list.append(cost)
    cost_mean += cost
    # Evaluate baseline
    baseline = cost_mean / (it + 1)

    # update policy
    grad = grad_traj(traj, u, gamma)
    gamma = gamma - lr * (cost - baseline) * grad
    if it % 10000 == 0:
        print("The current policy is " + str(gamma))


# Plot the smoothed cost curve according to learing iterations
smooth_cost_list = []
N_plot = np.int64(N_episodes / 10)
for i in range(N_plot):
    smooth_cost_list.append(np.mean(cost_list[i * 10 : (i + 1) * 10]))

plt.figure()
plt.plot(np.linspace(1, N_episodes, N_plot), np.array(smooth_cost_list))
plt.axis([1, N_episodes, 0, 10])
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Cost")
plt.show()
