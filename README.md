# CS-GY 6953 Project

Neural ODEs for actuated dynamical systems

### Create environment

```
conda env create -f environment.yml
conda activate node
```


## Part 1: Comparing continuous and discrete model

### Create datasets

```
python pendulum.py 
python cartpole.py 
```

### Train both models and plot results


```
python node.py 
python res_model.py
python plot.py 
```

## Part 2: Reinforcement learning

### Model based RL


```
python model_based_rl.py
```

### Reinforce algorithm

```
python reinforce.py
```