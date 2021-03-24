import numpy as np


def iterative_policy_eval(gridsize=(4,4), reward=-1., discountr=1., tolerance=1.0e-3, max_iter=1000, print_iter=False, print_final=False, policy="equiprob"):
    # Environment
    (m,n) = gridsize

    # Actions (0=Left,1=Right,2=Up,3=Down)
    actions = np.array([0,1,2,3])

    # State transition function (from state, action to next state)
    def transf(state, action):
        (row,col) = state
        if action == 0:
            if col==0: return (row,col)
            else: return (row,col-1)
        if action == 1:
            if col==3: return (row,col)
            else: return (row,col+1)
        if action == 2:
            if row==0: return (row,col)
            else: return (row-1,col)
        if action == 3:
            if row==3: return (row,col)
            else: return (row+1,col)

    """
    for i in range (0,m):
        for j in range (0,n):
            for a in actions:
                print(i,j,a)
                print(transf((i,j),a))
    """

    # Policy function, a m*n*a matrix with probability distribution of actions
    policyf = np.zeros((m,n,actions.size))
    if policy == "equiprob":
        policyf[:][:][:] = 1/actions.size
    else:
        raise ValueError("Enter correct policy type: equiprob.")
    best_policy = np.zeros((m,n,actions.size))

    # State and action value matrices
    statevals = np.zeros((m,n))
    newstatevals = np.zeros((m,n))
    actionvals = np.zeros((m,n,actions.size))

    # Control variables
    err = tolerance * 2
    k = 0

    # Iterations
    while err > tolerance and k < max_iter:
        # Clear values
        newstatevals = np.zeros((m, n))
        actionvals = np.zeros((m, n, actions.size))
        best_policy = np.zeros((m, n, actions.size))
        err = 0

        # Iterate through all states
        for row in range(0,m):
            for col in range(0,n):
                # Terminal states
                if row==col==0 or row==col==3:
                    stateval = 0
                # Non-terminal states
                else:
                    stateval = 0
                    for a in actions:
                        #print("row=%d,col=%d,a=%a" % (row,col,a))
                        (nxt_row, nxt_col) = transf((row,col), a)
                        actionval = policyf[row][col][a] * (reward + discountr * statevals[nxt_row][nxt_col])
                        #print(actionval)
                        stateval += actionval
                        actionvals[row][col][a] = actionval

                # Calculate error
                err = max(err, np.abs(float(statevals[row][col])-float(stateval)))
                #print("err:", err)

                # Store value of the state in new state value matrix
                newstatevals[row][col] = stateval

                # Evaluate best action for the state
                #print("actionvals",actionvals)
                best_actionval = np.amax(actionvals[row][col])
                for a in actions:
                    if actionvals[row][col][a] == best_actionval: best_policy[row][col][a] = 1

        # Update entire state value matrix
        statevals = newstatevals

        # Print for each iteration
        if print_iter:
            print("Iteration: ", k)
            print("State values:\n", np.round(statevals,1))
            print("Optimal policy:\n", best_policy)
            print("Error: ", np.round(err,12))
            print("\n")

        k += 1

    # Print final results
    if print_final:
        print("Total iterations: ", k)
        print("State Values:\n", np.round(statevals, 1))
        print("Optimal policy:\n", best_policy)
        print("Error: ", np.round(err, 12))
        print("\n")

    return k,statevals,best_policy

# Basic
run_basic = 0
if run_basic:
    iterative_policy_eval(max_iter=10, print_iter=True)
    iterative_policy_eval(max_iter=1000, print_final=True)

# Different discount rates:
run_discounted = 1
if run_discounted:
    for gamma in [0.8,0.5,0.25,0.1]:
        print("Discount rate: ", gamma)
        k,statevals,best_policy = iterative_policy_eval(max_iter=100, print_final=True, discountr=gamma)
# Note: it takes less iterations for smaller discount rates

# Different rewards:
run_rewards = 0
if run_rewards:
    print("Testing different reinforcements: ")
    for r in [-0.5,-2,-5,-10]:
        print("Reward: ", r)
        iterative_policy_eval(max_iter=1000, print_final=True, reward=r)
# Note: it takes more iterations for more severe penalties

