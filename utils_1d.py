import cvxpy as cp 
import numpy as np 
import matplotlib.pyplot as plt

'''
This file contains some functions for the 1-d system. For example:
x_{t+1} = a*x_t + u_t+w_t
'''

# Solve the offline optimal control policy
def offline_opt(x0, q, a, T, w):
    x = cp.Variable(T+1)
    u = cp.Variable(T)

    objective = cp.Minimize(q * cp.sum_squares(x) + cp.sum_squares(u))

    constraints = [x[0] == x0] # initial state constraint

    # Dynamics
    for t in range(0, T):
        constraints.append( x[t+1] == a * x[t] + u[t] + w[t] )
        
    prob = cp.Problem(objective, constraints)

    # Try to solve
    #, solver=cp.GUROBI, BarQCPConvTol=1e-8
    result = prob.solve(verbose=False)
    if result is None:
        print('Something goes wrong!')
    return x.value, u.value, result

# Solve the offline best linear controller
def cost_linear(x0, q, T, a, w, k):
    x = np.zeros(T+1)
    x[0] = x0
    u = np.zeros(T)
    for i in range(T):
        u[i] = -k*x[i]
        x[i+1] = a*x[i] + u[i] + w[i]
    return q*np.linalg.norm(x)**2 + np.linalg.norm(u)**2

def best_linear(x0, q, T, a, w):
    # search in the stable linear controller space
    # -1 < a - k < 1, a - 1 < k < a + 1
    K = np.linspace(a-1, a+1, 1000)
    k_opt = 0
    cost_opt = np.inf
    for k in K:
        cost = cost_linear(x0, q, T, a, w, k)
        if cost < cost_opt:
            cost_opt = cost
            k_opt = k
    
    x = np.zeros(T+1)
    x[0] = x0
    u = np.zeros(T)
    for i in range(T):
        u[i] = -k_opt*x[i]
        x[i+1] = a*x[i] + u[i] + w[i]

    return k_opt, cost_opt, x, u

# Visualize some roll out
def vis(rollouts, names):
    # rollouts[i] = [x, u, w]
    plt.figure(figsize=(12,4*len(rollouts)))
    for index in range(len(rollouts)):
        plt.subplot(len(rollouts), 3, 3*index+1)
        plt.plot(rollouts[index][0])
        plt.title("x" + '(' + names[index] + ')')
        plt.subplot(len(rollouts), 3, 3*index+2)
        plt.plot(rollouts[index][1])
        plt.title("u" + '(' + names[index] + ')')
        plt.subplot(len(rollouts), 3, 3*index+3)
        plt.plot(rollouts[index][2])
        plt.title("w" + '(' + names[index] + ')')
    plt.tight_layout()
    plt.show()

# Example: cost(LC)/cost(OPT) is very large
if __name__ == '__main__':
    T = 200
    x0 = 0.0
    w = 2.0 * np.random.uniform(size=T) - 1.0
    q = 10.0
    a = 3

    x_opt, u_opt, loss_opt = offline_opt(x0, q, a, T, w)
    K_opt, loss_lc, x_lc, u_lc = best_linear(x0, q, T, a, w)
    print('cost(OPT) =', loss_opt)
    print('cost(LC) =', loss_lc)
    print('K* =', K_opt)
    print('cost(LC)/cost(OPT) =', loss_lc/loss_opt)

    vis([[x_opt, u_opt, w], [x_lc, u_lc, w]], ['OPT', 'LC'])
