# Gridworld-RL

These are two projects from my Artificial Intelligence (CSC-261) course at Grinnell, in which I built reinforcement learning algorithms and test them in a 4x4 gridworld environment.

The first is a basic iterative policy evaluation algorithm with an initial equiprobable random policy. The second one is an off-policy Q-learning algorithm. With my partner Riccardo Morri, we used different parameters such as learning rate and exploration rate, and examine the change in the agent's speed of learning. 

For the Q-learning algorithm, we ran 3 experiments with the following parameters: 
- Experiment 1: discountr=0.9, learningr=0.1, explorationr=0.25
- Experiment 2: discountr=0.9, learningr=1/T, explorationr=1/T
- Experiment 3: discountr=0.9, learningr=0.1, explorationr=1/T
In each experiment, we ran 5 independent simulations. The average reward per episode is as follows:

![Exp1](https://github.com/mingxuan-he/Gridworld-RL/blob/main/exp1.jpg)
![Exp2](https://github.com/mingxuan-he/Gridworld-RL/blob/main/exp2.jpg)
![Exp3](https://github.com/mingxuan-he/Gridworld-RL/blob/main/exp3.jpg)
