import numpy as np
import random
import matplotlib.pyplot as plt


def qlearning(explr, learningr, gridsize=(4, 4), discountr=.9, tolerance=1.0e-3, max_eps=100, max_step=100, print_step=False, print_eps=False):
    # Environment
    (m, n) = gridsize

    # Actions (0=Left,1=Right,2=Up,3=Down)
    actions = np.array([0, 1, 2, 3])

    # State transition function (from state, action to next state)
    def transf(state, action):
        (row, col) = state
        if action == 0:
            if col == 0:
                return (row, col)
            else:
                return (row, col - 1)
        if action == 1:
            if col == 3:
                return (row, col)
            else:
                return (row, col + 1)
        if action == 2:
            if row == 0:
                return (row, col)
            else:
                return (row - 1, col)
        if action == 3:
            if row == 3:
                return (row, col)
            else:
                return (row + 1, col)

    """
    for i in range (0,m):
        for j in range (0,n):
            for a in actions:
                print(i,j,a)
                print(transf((i,j),a))
    """

    # Initialization
    eps = 1
    T = 1
    err = tolerance * 2
    Qtable = np.zeros((m, n, actions.size))
    #Qtable = np.random.rand(m,n,actions.size)
    newQtable = np.zeros((m, n, actions.size))
    eps_reward = []

    while err > tolerance and eps < max_eps:
        # Episode starts
        currstate = (3, 0)
        totalrewards = 0
        err = 0
        # Step starts
        while currstate != (0, 0) and currstate != (3, 3) and totalrewards > -max_step:
            (row, col) = currstate
            # Choose action (whether to explore or exploit)
            random.seed()
            randnum = random.random()
            # Explore
            if randnum < explr(T):
                random.seed()
                curraction = random.choice(actions)
            # Exploit
            else:
                random.seed()
                curraction = random.choice(np.flatnonzero(Qtable[row][col] ==Qtable[row][col].max()))

            # Next state and rewards
            nextstate = transf(currstate, curraction)
            (nextrow, nextcol) = nextstate
            if nextstate == (0, 0) or nextstate == (3, 3):
                reward = 1
            else:
                reward = -1
            totalrewards += reward

            # Calculate q value
            qmod = learningr(T) * (reward + discountr * np.amax(Qtable[nextrow][nextcol]) - Qtable[row][col][curraction])
            newQtable[row][col][curraction] += qmod

            # Calculate error
            err = max(err, np.abs(qmod))

            if print_step:
                print("T",T)
                print("state", currstate)
                print("action", curraction)
                print("lr",learningr(T))
                print("reward",reward)
                print("Q before", Qtable[row][col][curraction])
                print("Q after", newQtable[row][col][curraction])
                print("Error:", err)

            currstate = nextstate
            T += 1
            # Step ends

        Qtable[:][:][:] = newQtable[:][:][:]
        eps_reward.append(totalrewards)

        if print_eps:
            print("Episode ",eps)
            print(Qtable)
        eps += 1

        # Episode ends


    return eps, T, eps_reward, Qtable


def fixed_explr(T):
    return 0.25

def adaptive_explr(T):
    return 1 / T

def fixed_learningr(T):
    return 0.1

def adaptive_learningr(T):
    return 1 / T


def avg_reward(lst):
    row_lengths = []
    for row in lst:
        row_lengths.append(len(row))
    for row in lst:
        while len(row)<max(row_lengths):
            row.append(np.nan)

    list_avg = np.nanmean(np.array(lst), axis=0)
    return list_avg


run_exp_1 = 1
run_exp_2 = 1
run_exp_3 = 1

# Experiment 1
if run_exp_1:
    print("Experiment 1")
    rewards1 = []
    for i in range(5):
        e, t, R, Q = qlearning(explr=fixed_explr, learningr=fixed_learningr, discountr=0.9)
        rewards1.append(R)
        print("Experiment 1 Simulation ", i+1)
        print("Number of episodes",e)
        print("Rewards\n",R)
    avg_rewards1 = avg_reward(rewards1)
    #print(avg_rewards1)
    plt.plot(avg_rewards1)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Experiment 1")
    plt.axhline(y=-1, linestyle="dashed", label="optimality")
    plt.legend()
    plt.show()

# Experiment 2
if run_exp_2:
    print("Experiment 2")
    rewards2 = []
    for i in range(5):
        e, t, R, Q = qlearning(explr=adaptive_explr, learningr=adaptive_learningr, discountr=0.9, print_eps=False, print_step=False)
        rewards2.append(R)
        print("Experiment 2 Simulation ", i+1)
        print("Number of episodes:", e)
        print("Rewards\n", R)
    avg_rewards2 = avg_reward(rewards2)
    #print(avg_rewards2)
    plt.plot(avg_rewards2)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Experiment 2")
    plt.axhline(y=-1, linestyle="dashed", label="optimality")
    plt.legend()
    plt.show()



# Experiment 3
if run_exp_3:
    print("Experiment 3")
    rewards3 = []
    for i in range(5):
        e, t, R, Q = qlearning(explr=adaptive_explr, learningr=fixed_learningr, discountr=0.9,print_eps=False, print_step=False)
        rewards3.append(R)
        print("Experiment 3 Simulation ", i+1)
        print("Number of episodes:", e)
        print("Rewards:\n", R)
    #print(rewards3)
    avg_rewards3 = avg_reward(rewards3)
    #print(avg_rewards3)
    plt.plot(avg_rewards3)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Experiment 3")
    plt.axhline(y=-1, linestyle="dashed", label="optimality")
    plt.legend()
    plt.show()




