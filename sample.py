import numpy as np
from numba import jit
from network_utilities import * 
from collections import defaultdict


def scalability_classes(A, method):
    """returns a dictionary with the scalability classes, 
    meaning the unique classes of coupled in and out degree
    """
    if method == 'dcm_rd':
        k_out = out_degree(A)
        k_in = in_degree(A)
        d = defaultdict(list)
        tup = tuple([k_out[0], k_in[0]])
        d[tup] = [0]
        n = len(k_out)
        for i in range(1, n):
            # visit each couple (in, out) and add new classes to the dict
            tup = tuple([k_out[i], k_in[i]])
            d[tup].append(i)
        return d


def setup(A, method):
    """takes in input adjacency matrix and method string 
    and returns the parameters array and the initial point
    """
    if method == 'dcm':
        k_out = out_degree(A)
        k_in = in_degree(A)
        par = np.concatenate((k_out, k_in))
        L = A.sum()
        # starting point
        x = k_out/np.sqrt(L)
        y = k_in/np.sqrt(L)
        v0 = np.concatenate((x, y))
        
        return [par, v0]

    if method == 'dcm_rd':
        d = scalability_classes(A, method='dcm_rd')
        keys = list(d.keys())
        k_out = np.array([x[0] for x in keys])
        k_in = np.array([x[1] for x in keys])
        c = np.array([len(x) for x in list(d.values())])
        par = np.concatenate((k_out, k_in, c))
        # starting point
        L = A.sum()
        x = k_out/np.sqrt(L)
        y = k_in/np.sqrt(L)
        v0 = np.concatenate((x, y))

        return [par, v0]


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


@jit(nopython=True)
def iterative_fun_dcm_rd(v, par):
    """
    :param v: np.array
    :param par: np.array
    :return: scalar float

    """

    n = int(len(v)/2)
    x = v[0:n]
    y = v[n:2*n]
    k_out = par[0:n]
    k_in = par[n:2*n]
    c = par[2*n:3*n]

    xd = np.zeros(n)
    yd = np.zeros(n)

    for i in range(0, n):
        for j in range(0, n):
            if j != i:
                xd[i] += c[j]*y[j]/(1 + x[i]*y[j])
                yd[i] += c[j]*x[j]/(1 + y[i]*x[j])
            else:
                xd[i] += (c[i] - 1)*y[i]/(1 + x[i]*y[i])
                yd[i] += (c[i] - 1)*x[i]/(1 + x[i]*y[i])

    # calculate final solutions xx and yy
    xx = k_out/xd
    yy = k_in/yd

    return np.concatenate((xx, yy))


def iterative_solver(A, max_steps = 300, eps = 0.01, method = 'dcm'):
    """Solve the DCM problem of the network

    INPUT:
        * A: adjacency matrix 
        * max_steps: maximum number of steps
        * method: 'dcm', 'dcm_rd'
    OUTPUT:
        * [x, y] parameters solutions
    """
    # method choice
    f_dict = {
            'dcm' : iterative_fun_dcm,
            'dcm_rd': iterative_fun_dcm_rd
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
