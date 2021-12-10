import numpy as np
import matplotlib.pyplot as plt


# Choose experiment: Pendulum or Cartpole
name = "Cartpole"

resnet_loss = np.load("log/" + name + "resnet_loss.npy")
node_loss = np.load("log/" + name + "node_loss.npy")

mean_node = np.mean(node_loss, axis=0)
std_node = np.std(node_loss, axis=0)

mean_res = np.mean(resnet_loss, axis=0)
std_res = np.std(resnet_loss, axis=0)

x = np.linspace(1, len(mean_node), len(mean_node))

plt.figure()
plt.fill_between(x, mean_res - std_res, mean_res + std_res, color="lightskyblue")
plt.fill_between(x, mean_node - std_node, mean_node + std_node, color="orange")
plt.plot(x, mean_res, color="blue", label="Discrete model")
plt.plot(x, mean_node, color="darkorange", label="Continuous model")
plt.xlim(x[0], x[-1])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.fill_between(x, mean_res - std_res, mean_res + std_res, color="lightskyblue")
plt.fill_between(x, mean_node - std_node, mean_node + std_node, color="orange")
plt.plot(x, mean_res, color="blue", label="Discrete model")
plt.plot(x, mean_node, color="darkorange", label="Continuous model")
plt.xlim(x[0], x[-1])
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
