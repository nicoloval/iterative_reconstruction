import numpy as np
from numba import jit
from network_utilities import * 

@jit(nopython=True)
def iterative_fun(x, y, k_out, k_in):
    """Return the next iterative step.
    All inputs should have the same dimension

    Input:
        * x at step n
        * y at step n
        * k_in
        * k_out
    Output:
        * [x, y] at step n+1
    
    """
    # problem dimension
    n = len(x)

    # calculate the denominators 
    xd = np.zeros(n)
    yd = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                xd[i] += y[j]/(1 + x[i]*y[j])
                yd[i] += x[j]/(1 + y[i]*x[j])

    # calculate final solutions xx and yy
    xx = k_out/xd
    yy = k_in/yd

    return xx, yy


def iterative_solver(A, max_steps = 300, eps = 0.01):
    """Solve the DCM problem of the network

    INPUT:
        * A: adjacency matrix 
        * max_steps: maximum number of steps
    OUTPUT:
        * [x, y] parameters solutions
    """
    k_out = out_degree(A)
    k_in = in_degree(A)
    L = A.sum()
    # starting point
    x = k_out/np.sqrt(L)
    y = k_in/np.sqrt(L)
    
    step = 0
    diff = eps + 1
    while diff > eps and step < max_steps:
        # iterative step
        xx, yy = iterative_fun(x, y, k_out, k_in)
        # convergence step
        n = np.concatenate((x,y))
        nn = np.concatenate((xx,yy))
        diff = np.linalg.norm(n - nn)/np.linalg.norm(n)  # 2-norm 
        del n, nn
        # set next step
        x = xx
        y = yy
        del xx, yy
        step += 1
    # output  
    sol = np.concatenate((x, y))

    return sol, step, diff
