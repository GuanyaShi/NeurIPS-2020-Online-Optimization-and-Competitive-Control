import cvxpy as cp 
import numpy as np 
import matplotlib.pyplot as plt

'''
This file contains some functions for the 2-d SIMO system. For example:
A = [0  1; -1 2]
B = [0; 1]
x_{t+1} = A*x_t + B*(u_t+w_t)
'''

# Compute the cost of some control sequence
def eva(Q, x, u, R=1):
    return Q*np.linalg.norm(x)**2 + R*np.linalg.norm(u)**2

# Solve the offlineoptimal control policy
def offline_opt(x0, Q, A, B, T, w, R=1):
    x = cp.Variable((len(x0), T+1))
    u = cp.Variable((1, T))

    objective = cp.Minimize(Q * cp.sum_squares(x) + R * cp.sum_squares(u))

    constraints = [x[:, 0] == x0] # initial state constraint

    # Dynamics
    for t in range(0, T):
        constraints.append( x[:, t+1] == A @ x[:, t] + B @ u[:, t] + B @ w[:, t] )

    prob = cp.Problem(objective, constraints)

    # Try to solve
    #result = prob.solve(verbose=False, solver=cp.GUROBI, BarQCPConvTol=1e-8)
    result = prob.solve(verbose=False)
    if result is None:
        print('Something goes wrong!')
        
    return x.value, u.value, eva(Q, x.value, u.value)

# Roll out linear controller (u_t = K * x_t) and compute the cost
def linear_con(Q, A, B, x0, K, w, T, R=1):
    x = np.zeros((2, T+1))
    u = np.zeros((1, T))
    x[:, 0] = x0
    for t in range(0, T):
        u[:, t] = K @ x[:, t]
        x[:, t+1] = A @ x[:, t] + B @ u[:, t] + B @ w[:, t]
    return x, u, eva(Q, x, u)

# Visualize some roll out
def vis(rollouts, names):
    # rollouts[i] = [x, u, w]
    plt.figure(figsize=(12,4*len(rollouts)))
    for index in range(len(rollouts)):
        plt.subplot(len(rollouts), 3, 3*index+1)
        plt.plot(rollouts[index][0].transpose())
        plt.legend(['x1', 'x2'])
        plt.title("x" + '(' + names[index] + ')')
        plt.subplot(len(rollouts), 3, 3*index+2)
        plt.plot(rollouts[index][1].transpose())
        plt.title("u" + '(' + names[index] + ')')
        plt.subplot(len(rollouts), 3, 3*index+3)
        plt.plot(rollouts[index][2].transpose())
        plt.title("w" + '(' + names[index] + ')')
    plt.tight_layout()
    plt.show()

# Search the best linear controller
# We just need to search in the stable controller space \mathcal{K}. 
# For A = [0  1; -1 2], B = [0; 1], \mathcal{K} \in {K | [-0.1 2.1] < K < [-4.1 0.1]}  
def search_linear(Q, A, B, x0, w, T, R=1):
    loss_lin_opt = np.inf
    K_opt = np.nan
    
    a = np.linspace(-0.1, 2.1, 80) # k1
    b = np.linspace(-4.1, 0.1, 80) # k2
    for j in range(len(a)):
        for i in range(len(b)):
            K = np.array([[a[j], b[i]]])
            r = np.max(np.abs(np.linalg.eigvals(A + B @ K)))
            if r < 1:
                x_lin, u_lin, loss_lin = linear_con(Q, A, B, x0, K, w, T)
                if loss_lin < loss_lin_opt:
                    loss_lin_opt = loss_lin
                    K_opt = K
            
    K_ref = K_opt
    # Search in K_ref + [-0.05, 0.05] x [-0.05, 0.05]
    for i in range(20):
        for j in range(20):
            dK = np.array([[0.005*i-0.05, 0.005*j-0.05]])
            x_lin, u_lin, loss_lin = linear_con(Q, A, B, x0, K_ref+dK, w, T)
            if loss_lin < loss_lin_opt:
                loss_lin_opt = loss_lin
                K_opt = K_ref+dK
    
    x_lc, u_lc, loss_lc = linear_con(Q, A, B, x0, K_opt, w, T)

    return x_lc, u_lc, loss_lc, K_opt

# Example: cost(LC)/cost(OPT) is very large
if __name__ == '__main__':
    T = 200
    x0 = np.array([0.0, 0.0])
    w = 2 * (np.random.uniform(size=[1, T]) - 0.5)
    Q = 10.0
    A = np.array([[0.0, 1.0], [-1.0, 2.0]])
    B = np.array([[0.0], [1.0]])

    x_opt, u_opt, loss_opt = offline_opt(x0, Q, A, B, T, w)
    x_lc, u_lc, loss_lc, K_opt = search_linear(Q, A, B, x0, w, T)
    print('cost(OPT) =', loss_opt)
    print('cost(LC) =', loss_lc)
    print('K* =', K_opt)
    print('cost(LC)/cost(OPT) =', loss_lc/loss_opt)

    vis([[x_opt, u_opt, w], [x_lc, u_lc, w]], ['OPT', 'LC'])
