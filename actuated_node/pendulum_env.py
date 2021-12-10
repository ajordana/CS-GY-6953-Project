import numpy as np
import matplotlib.pyplot as plt


class Pendulum:
    """
    Environment class for the simple pendulum
    """

    def __init__(self):
        super(Pendulum, self).__init__()
        self.max_torque = 2
        self.dt = 0.1
        self.m = 1
        self.g = 9.81
        self.l = 1.0

    def f(self, state, u):
        th, thdot = state
        return np.array([thdot, -self.g * np.sin(th) + u])

    def step(self, state, u):
        ## RK4 step
        k1 = self.f(state, u) * self.dt
        k2 = self.f(state + k1 / 2.0, u) * self.dt
        k3 = self.f(state + k2 / 2.0, u) * self.dt
        k4 = self.f(state + k3, u) * self.dt
        state = state + (k1 + 2 * (k2 + k3) + k4) / 6
        return state


if __name__ == "__main__":
    # Example of with a PD controller
    env = Pendulum()
    state = np.zeros(2)
    Kp = 16
    Kd = 8
    T = 50
    traj = np.zeros((T, 2))
    for i in range(T - 1):
        tau = -Kp * (state[0] - np.pi) - Kd * state[1]
        state = env.step(state, tau)
        traj[i + 1] = state
    time = np.linspace(0, 5, T)
    plt.plot(time, traj[:, 0], label="th")
    plt.plot(time, traj[:, 1], label="th_dot")
    plt.plot(time[-1], np.pi, "o")
    plt.legend()
    plt.show()
