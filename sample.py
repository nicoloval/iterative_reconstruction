import numpy as np
from numba import jit
from network_utilities import * 

@jit(nopython=True)
def iterative_fun_dcm(v, par):
    """Return the next iterative step.
    All inputs should have the same dimension

    Input:
        * (x, y) at step n
        * (k_out, k_in)
    Output:
        * (x, y) at step n+1
    
    """
    # problem dimension
    n = int(len(v)/2)
    x = v[0:n]
    y = v[n:2*n]
    k_out = par[0:n]
    k_in = par[n:2*n]
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

    return np.concatenate((xx, yy))


def iterative_solver(A, max_steps = 300, eps = 0.01, method = 'dcm'):
    """Solve the DCM problem of the network

    INPUT:
        * A: adjacency matrix 
        * max_steps: maximum number of steps
    OUTPUT:
        * [x, y] parameters solutions
    """
    # method choice
    f_dict = {
            'dcm' : iterative_fun_dcm
            }
    iterative_fun = f_dict[method]
    # 
    k_out = out_degree(A)
    k_in = in_degree(A)
    par = np.concatenate((k_out, k_in))
    L = A.sum()
    # starting point
    x = k_out/np.sqrt(L)
    y = k_in/np.sqrt(L)
    v = np.concatenate((x, y))

    step = 0
    diff = eps + 1
    while diff > eps and step < max_steps:
        # iterative step
        vv = iterative_fun(v, par)
        # convergence step
        diff = np.linalg.norm(v - vv)/np.linalg.norm(v)  # 2-norm 
        del v
        # set next step
        v = vv
        del vv
        step += 1
    # output  
    sol = v

    return sol, step, diff
