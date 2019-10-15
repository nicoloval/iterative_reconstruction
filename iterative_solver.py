import numpy as np
from numba import jit

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
            if i != j:
                xd[i] += xd[i] + y[j]/(1 + x[i]*y[j])

    # calculate final solutions xx and yy
    xx = k_out/xd
    yy = k_in/yd

    return [xx, yy]

