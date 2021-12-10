import numpy as np
import matplotlib.pyplot as plt
from utils import makedirs


class Pendulum:
    """
    Environment class for the simple pendulum
    """

    def __init__(self):
        super(Pendulum, self).__init__()
        self.max_torque = 2
        self.dt = 0.001
        self.m = 1
        self.g = 9.81
        self.l = 1.0

    def f(self, state, u):
        th, thdot = state
        return np.array([thdot, -self.g * np.sin(th) + u])

    def step(self, u):
        ## RK4 step
        k1 = self.f(self.state, u) * self.dt
        k2 = self.f(self.state + k1 / 2.0, u) * self.dt
        k3 = self.f(self.state + k2 / 2.0, u) * self.dt
        k4 = self.f(self.state + k3, u) * self.dt
        self.state = self.state + (k1 + 2 * (k2 + k3) + k4) / 6
        return self.state.copy()

    def reset(self):
        high = np.array([np.pi, 1.0])
        r1, r2 = 2 * np.random.rand(2) - 1
        self.state = np.array([r1 * high[0], r2 * high[1]])
        return self.state.copy()

    def sample_action(self):
        r = 2 * np.random.rand(1) - 1
        return r[0] * self.max_torque

    def EM(self, state):
        th, thdot = state
        return self.g * (1 - np.cos(th)) + 0.5 * thdot ** 2 > 2 * self.g


if __name__ == "__main__":

    makedirs("datasets")

    # Initialize environment
    model = Pendulum()
    dt = model.dt
    Dt = 1e-1

    # Collect training set
    n_sim_train = 1000
    T = 1
    N_sim = int(T / model.dt)
    N = int(T / Dt)
    fac = np.int64(np.round(N_sim / N))
    record_train = np.zeros((n_sim_train, N, 2), dtype=np.float32)
    u = 0.0
    sim = 0
    while sim < n_sim_train:
        x = model.reset()
        if not model.EM(x):
            for i in range(N_sim):
                x = model.step(u)
                if i % fac == 0:
                    index = i // fac
                    record_train[sim, index] = x
        sim += 1
    np.save("datasets/Pendulum_train", record_train)
    time = np.linspace(0.0, T, N)
    np.save("datasets/Pendulum_time_train", time)

    # Collect testing set
    n_sim_test = 1000
    T = 1
    N_sim = int(T / model.dt)
    N = int(T / Dt)

    record_test = np.zeros((n_sim_test, N, 2), dtype=np.float32)
    u = 0.0
    sim = 0
    while sim < n_sim_test:
        x = model.reset()
        if not model.EM(x):
            for i in range(N_sim):
                x = model.step(u)
                if i % fac == 0:
                    index = i // fac
                    record_test[sim, index] = x
            sim += 1
    np.save("datasets/Pendulum_test", record_test)

    time = np.linspace(0.0, T, N)
    np.save("datasets/Pendulum_time_test", time)

    # plot a test trajectory
    index = 0
    a = record_test[index]
    plt.figure(0)
    plt.plot(a[:, 0], "o")
    plt.plot(a[:, 1], "x")
    plt.show()
