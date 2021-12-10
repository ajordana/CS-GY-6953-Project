import numpy as np
import matplotlib.pyplot as plt
from utils import makedirs


class CartPole:
    """
    Environment class for the inverted cart CartPole
    """

    def __init__(self):
        super(CartPole, self).__init__()
        self.max_control = 20
        self.dt = 0.001
        self.m = 1.0
        self.M = 1.0
        self.g = 9.81
        self.L = 1.0
        self.d = 1.0

    def f(self, x, u):
        Sy = np.sin(x[2])
        Cy = np.cos(x[2])
        D = self.m * self.L ** 2 * (self.M + self.m * (1 - Cy ** 2))
        dx = [0] * 4
        dx[0] = x[1]
        dx[1] = (1 / D) * (
            -self.m ** 2 * self.L ** 2 * (-self.g) * Cy * Sy
            + self.m * self.L ** 2 * (self.m * self.L * x[3]) ** 2 * Sy
            - self.d * x[1]
        ) + self.m * self.L ** 2 * (1 / D) * u
        dx[2] = x[3]
        dx[3] = (1 / D) * (
            (self.m + self.M) * self.m * (-self.g) * self.L * Sy
            - self.m * self.L * Cy * (self.m * self.L * x[3] ** 2 * Sy - self.d * x[1])
        ) - self.m * self.L * Cy * (1 / D) * u
        return np.array(dx)

    def step(self, u):  ## RK4 step
        k1 = self.f(self.state, u) * self.dt
        k2 = self.f(self.state + k1 / 2.0, u) * self.dt
        k3 = self.f(self.state + k2 / 2.0, u) * self.dt
        k4 = self.f(self.state + k3, u) * self.dt
        self.state = self.state + (k1 + 2 * (k2 + k3) + k4) / 6
        return self.state.copy()

    def reset(self):
        high = np.array([4.0, 2.0, np.pi, 4.0])
        r1, r2, r3, r4 = 2 * np.random.rand(4) - 1
        self.state = np.array([r1 * high[0], r2 * high[1], r3 * high[2], r4 * high[3]])
        return self.state.copy()

    def sample_action(self):
        r = 2 * np.random.rand(1) - 1
        return r[0] * self.max_control


if __name__ == "__main__":

    makedirs("datasets")

    # Initialize environment
    model = CartPole()
    dt = model.dt
    Dt = 1e-1

    # Collect training set
    n_sim_train = 4000
    T = 1
    N_sim = int(T / model.dt)
    N = int(T / Dt)
    fac = np.int64(np.round(N_sim / N))
    record_train = np.zeros((n_sim_train, N, 4), dtype=np.float32)
    u = 0.0
    sim = 0
    while sim < n_sim_train:
        x = model.reset()
        for i in range(N_sim):
            x = model.step(u)
            if i % fac == 0:
                index = i // fac
                record_train[sim, index] = x
        sim += 1
    np.save("datasets/Cartpole_train", record_train)
    time = np.linspace(0.0, T, N)
    np.save("datasets/Cartpole_time_train", time)

    # Collect testing set
    n_sim_test = 1000
    T = 1
    N_sim = int(T / model.dt)
    N = int(T / Dt)

    record_test = np.zeros((n_sim_test, N, 4), dtype=np.float32)
    u = 0.0
    sim = 0
    while sim < n_sim_test:
        x = model.reset()
        for i in range(N_sim):
            x = model.step(u)
            if i % fac == 0:
                index = i // fac
                record_test[sim, index] = x
        sim += 1
    np.save("datasets/Cartpole_test", record_test)

    time = np.linspace(0.0, T, N)
    np.save("datasets/Cartpole_time_test", time)

    # plot a test trajectory
    index = 0
    a = record_test[index]
    plt.figure()
    plt.plot(a[:, 0], "o")
    plt.plot(a[:, 1], "x")
    plt.plot(a[:, 2], "--")
    plt.plot(a[:, 3], "-")
    plt.show()
