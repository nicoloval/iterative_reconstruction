import numpy as np
from numba import jit
from collections import OrderedDict
import scipy.sparse


class OrderedDefaultListDict(OrderedDict): #name according to default
    def __missing__(self, key):
        self[key] = value = [] #change to whatever default you want
        return value


def scalability_classes(A, method):
    """returns a dictionary with the scalability classes, 
    meaning the unique classes of coupled in and out degree
    """
    if method == 'dcm_rd':
        k_out = out_degree(A)
        k_in = in_degree(A)
        d = OrderedDefaultListDict()
        tup = tuple([k_out[0], k_in[0]])
        d[tup] = [0]
        n = len(k_out)
        for i in range(1, n):
            # visit each couple (in, out) and add new classes to the dict
            tup = tuple([k_out[i], k_in[i]])
            d[tup].append(i)
        return d


def rd2full(x, d, method):
    """converts a reduced vector to full form
    """
    if method == 'dcm_rd':
        return rd2full_dcm_rd(x, d)


def rd2full_dcm_rd(x, d):
    val = list(d.values())
    n = 0  # dimension of the full solution
    m = len(val)
    for i in range(0, m):  # todo: usare un sum invece del ciclo
        n += len(val[i])
    # allocate full solution
    y1 = np.zeros(n, dtype=x.dtype)
    y2 = np.zeros(n, dtype=x.dtype)
    for i in range(0, m): 
        y1[val[i]] = x[i]
        y2[val[i]] = x[i+m]
    return np.hstack((y1, y2))


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
        keys = list(d)
        k_out = np.array([x[0] for x in keys])
        k_in = np.array([x[1] for x in keys])
        c = np.array([len(d[(kout,kin)]) for kout,kin in zip(k_out, k_in)])
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
    # function choice
    f_dict = {
            'dcm' : iterative_fun_dcm,
            'dcm_rd': iterative_fun_dcm_rd
            }
    iterative_fun = f_dict[method]

    # initial setup
    par, v = setup(A, method)

    # iteration steps
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


def out_degree(a):
    # todo: for out_degree and in_degree...check np.int32 is always returned
    """returns matrix A out degrees

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1


def in_degree(a):
    """returns matrix A in degrees

    :param a: np.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 0).A1


def expected_out_degree(sol, method, d=None):
    # TODO: controllare che funzioni
    """returns expected out degree after ERGM method, on undirected networks
    returns just the expected degree

    Parameters
    ----------

    :param sol: :class:`~numpy.ndarray`
        Solution of the ERGM problem

    Returns
    -------

    :return k: :class:`~numpy.ndarray`
        array of expected out-degrees

    """
    if method == 'dcm':
        return expected_out_degree_dcm(sol)

    if method == 'dcm_rd':
        # cardinality of reduced equivalent classes
        c = [len(d[key]) for key in d.keys()]  
        k = expected_out_degree_dcm_rd(sol, c)
        # convert rd to full array
        m = len(d)
        d_vals = list(d.values())
        n = np.array([len(d[x]) for x in d]).sum()
        y = np.zeros(n, dtype=k.dtype)
        for i in range(m):
            y[d_vals[i]] = k[i] 

        return y 


        return rd2full_dcm_rd(k, d)


@jit(nopython=True)
def expected_out_degree_dcm(sol):
    n = int(sol.size / 2)
    a_out = sol[0:n]
    a_in = sol[n:]

    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a_in[j]*a_out[i] / (1 + a_in[j]*a_out[i])

    return k


@jit(nopython=True)
def expected_out_degree_dcm_rd(sol, c):
    n = int(sol.size/2)
    a_out = sol[0:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k

    for i in range(n):
        for j in range(n):
            if j != i:
                k[i] += c[j]*a_in[j]*a_out[i]/(1 + a_in[j]*a_out[i])
            else:
                k[i] += (c[i] - 1)*a_in[i]*a_out[i]/(1 + a_in[i]*a_out[i])
    return k 


def expected_in_degree(sol, method, d=None):
    """returns expected in degree after ERGM method, on undirected networks
    returns just the expected degree

    Parameters
    ----------

    :param sol: :class:`~numpy.ndarray`
        Solution of the ERGM problem

    Returns
    -------

    :return k: :class:`~numpy.ndarray`
        array of expected in-degrees
    """
    if method == 'dcm':
        return expected_in_degree_dcm(sol)

    if method == 'dcm_rd':
        # cardinality of scalability classes 
        c = [len(d[key]) for key in d.keys()]
        # expected in degree by class
        k = expected_in_degree_dcm_rd(sol, c)
        # convert rd to full array
        m = len(d)
        d_vals = list(d.values())
        n = np.array([len(d[x]) for x in d]).sum()
        y = np.zeros(n, dtype=k.dtype)
        for i in range(m):
            y[d_vals[i]] = k[i] 

        return y 


@jit(nopython=True)
def expected_in_degree_dcm(sol):
    n = int(sol.size/2)
    a_out = sol[0:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a_in[i]*a_out[j]/(1 + a_in[i]*a_out[j])

    return k


@jit(nopython=True)
def expected_in_degree_dcm_rd(sol, c):
    n = int(sol.size/2)
    a_out = sol[0:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k

    for i in range(n):
        for j in range(n):
            if j != i:
                k[i] += c[j]*a_out[j]*a_in[i]/(1 + a_out[j]*a_in[i])
            else:
                k[i] += (c[i] - 1)*a_out[i]*a_in[i]/(1 + a_out[i]*a_in[i])

    return k 


def expected_dyads(sol, method, A=None, t="dyads"):
    """
    Computes the expected number of dyads on the ERGM ensemble
    :param sol: np.ndarray, problem's solution 
    :param method: string, the available ERGM methods:
        'dcm':
        'dcm_rd':
    :param d: ordered Dict, contains the info about the reduced system
    :return:
    """
    if method == 'dcm':
        if t == 'dyads':
            return expected_dyads_dcm(sol)
        if t == 'singles':
            return expected_singles_dcm(sol)
        if t == 'zeros':
            return expected_zeros_dcm(sol)

    if method == 'dcm_rd':
        d = scalability_classes(A, 'dcm_rd')
        sol_full = rd2full(sol, d, 'dcm_rd')
 
        if t == 'dyads':
            return expected_dyads_dcm(sol_full)
        if t == 'singles':
            return expected_singles_dcm(sol_full)
        if t == 'zeros':
            return expected_zeros_dcm(sol_full)

        #TODO check the following commented code and dcm_rd method for dyads
        """ 
        # cardinality of scalability classes 
        c = [len(d[key]) for key in d.keys()]
        # expected in degree by class
        ed = expected_dyads_dcm_rd(sol, c)
        # convert rd to full array
        m = len(d)
        d_vals = list(d.values())
        n = np.array([len(d[x]) for x in d]).sum()
        y = np.zeros(n, dtype=ed.dtype)
        for i in range(m):
            y[d_vals[i]] = ed[i]
        return y
        return y
        """


@jit(nopython=True)
def expected_dyads_dcm(sol):
    """ compute the expected number of reciprocated links 
    """
    # edges
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += x[j]*y[j]/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= x[i]*y[i]/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += x[i]*y[i]*temp
    return er


@jit(nopython=True)
def expected_singles_dcm(sol):
    """ compute the expected number of non reciprocated links 
    """
    # edges
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += (y[i]*x[j] + y[j]*x[i])/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= 2*x[i]*y[i]/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += temp
    return er


@jit(nopython=True)
def expected_zeros_dcm(sol):
    """ compute the expected number of non present links (number of couples not connected)
    """
    # edges
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += 1/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= 1/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += temp
    return er


@jit(nopython=True)
def expected_dyads_dcm_rd(sol, c):
    #TODO: redefine this function in a working way
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            if i != j:
                temp += c[j]*x[j]*y[j]/((1 + x[i]*y[j])*(1 + x[j]*y[i]))
            else:
                temp += (c[j] - 1)*x[j]*y[j] / \
                        ((1 + x[i]*y[j])*(1 + x[j]*y[i]))
        er += c[i]*x[i]*y[i]*temp
    return er


