import numpy as np
import os
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
    if method == 'cm':
        # A should be symmetric!!!
        par = out_degree(A)
        L = A.sum()
        # starting point
        v0 = par/np.sqrt(L)

        return [par, v0]

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

    if method == 'rdcm':
        k_out_nr = non_reciprocated_out_degree(A)
        k_in_nr = non_reciprocated_in_degree(A)
        k_r = reciprocated_degree(A)
        par = np.concatenate((k_out_nr, k_in_nr, k_r))
        L = A.sum()
        # starting point
        x = k_out_nr/np.sqrt(L)
        y = k_in_nr/np.sqrt(L)
        z = k_r/np.sqrt(L)
        v0 = np.concatenate((x, y, z))
        
        return [par, v0]




@jit(nopython=True)
def iterative_fun_cm(v, par):
    """Return the next iterative step.
    All inputs should have the same dimension

    Input:
    ------
        * x at step n
        * k
    Output:
    -------
        * x at step n+1
    """

    # problem dimension
    n = len(v)
    x = v
    k = par
    # calculate the denominators 
    xd = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if j != i:
                xd[i] += x[j]/(1 + x[i]*x[j])

    # calculate final solutions xx and yy
    xx = k/xd

    return xx 


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


@jit(nopython=True)
def iterative_fun_rdcm(v, par):
    """Return the next iterative step.
    All inputs should have the same dimension

    Input:
        * (x, y, z) at step n
        * (k_out_nr, k_in_nr, k_r)
    Output:
        * (x, y, z) at step n+1
    
    """
    # problem dimension
    n = int(len(v)/3)
    x = v[0:n]
    y = v[n:2*n]
    z = v[2*n:3*n]
    k_out_nr = par[0:n]
    k_in_nr = par[n:2*n]
    k_r = par[2*n:3*n]
    # calculate the denominators 
    xd = np.zeros(n)
    yd = np.zeros(n)
    zd = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if j != i:
                den = 1 + x[i]*y[j] + x[j]*y[i] \
                    + z[i]*z[j]
                xd[i] += y[j]/den
                yd[i] += x[j]/den
                zd[i] += z[j]/den

    # calculate final solutions xx and yy
    xx = k_out_nr/xd
    yy = k_in_nr/yd
    zz = k_r/zd

    return np.concatenate((xx, yy, zz))


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
            'cm' : iterative_fun_cm,
            'dcm' : iterative_fun_dcm,
            'dcm_rd': iterative_fun_dcm_rd,
            'rdcm': iterative_fun_rdcm
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


def non_reciprocated_out_degree(a):
    if type(a) == np.ndarray:
        s = a.shape
        one = np.ones(shape=s)
        return np.diagonal(a@(one - a))
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        s = a.shape
        one = np.ones(shape=s)
        o = scipy.sparse.csr_matrix(one)
        return (a.dot(o - a)).diagonal()


def non_reciprocated_in_degree(a):
    if type(a) == np.ndarray:
        s = a.shape
        one = np.ones(shape=s)
        return np.diagonal(a.transpose()@(one - a.transpose()))
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        s = a.shape
        one = np.ones(shape=s)
        o = scipy.sparse.csr_matrix(one)
        return (a.transpose().dot(o - a.transpose())).diagonal()


def reciprocated_degree(a):
    if type(a) == np.ndarray:
        return np.diagonal(a@a)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return (a.dot(a)).diagonal()


def dyads_count(a):
    """Counts number of dyads
    """
    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int(len(tmp[tmp == 2]))
    if isinstance(a, (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)):
        return int(tmp[tmp == 2].shape[1]) 


def singles_count(a):
    """Counts number of not-reciprocated links
    """
    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int(len(tmp[tmp == 1])/2)
    if isinstance(a, (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)):
        return int(tmp[tmp == 1].shape[1]/2) 


def zeros_count(a):
    """Counts number of empty couples of nodes
    """
    n = a.shape[0]
    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int((n*(n-1) - np.count_nonzero(tmp)))
    if isinstance(a, (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)):
        return int((n*(n-1) - tmp.count_nonzero()))


def expected_degree(sol, method, directed=None, d=None):
    """returns expected degree after ERGM method

    Parameters
    ----------

    :param sol: :class:`~numpy.ndarray`
        Solution of the ERGM problem
    :param method: :class:`~string`
        String stands for the requested method
    :param directed: :class:`~string`
        If the method is for directed network.
        Accepted values are "In" and "Out"
    :param d: :class:`~dict`
        Scalability map for reduced method


    Returns
    -------

    :return k: :class:`~numpy.ndarray`
        array of expected out-degrees
    """
    # undericted methods
    if method == 'cm':
        return expected_degree_cm(sol)
    # directed methods 
    if directed == 'Out':
        if method == 'dcm':
            return expected_out_degree_dcm(sol)
        if method == 'dcm_rd':
            sol_full = rd2full(sol, d, 'dcm_rd')
            return expected_out_degree_dcm(sol_full)
    if directed == 'In':
        if method == 'dcm':
            return expected_in_degree_dcm(sol)
        if method == 'dcm_rd':
            sol_full = rd2full(sol, d, 'dcm_rd')
            return expected_in_degree_dcm(sol_full)


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


@jit(nopython=True)
def expected_degree_cm(sol):
    n = len(sol)
    a = sol

    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a[j]*a[i] / (1 + a[j]*a[i])

    return k


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


@jit(nopython=True)
def expected_non_reciprocated_in_degree_rdcm(sol):
    n = int(sol.size/3)
    x = sol[0:n]
    y = sol[n:2*n]
    z = sol[2*n:3*n]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += x[j]*y[i]/(1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j])

    return k


@jit(nopython=True)
def expected_non_reciprocated_out_degree_rdcm(sol):
    n = int(sol.size/3)
    x = sol[0:n]
    y = sol[n:2*n]
    z = sol[2*n:3*n]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += x[i]*y[j]/(1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j])

    return k


@jit(nopython=True)
def expected_reciprocated_degree_rdcm(sol):
    n = int(sol.size/3)
    x = sol[0:n]
    y = sol[n:2*n]
    z = sol[2*n:3*n]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += z[i]*z[j]/(1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j])

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


def std_dyads(sol, method, A=None, t="dyads"):
    """
    Computes the standard deviation of the number of dyads on the ERGM ensemble
    :param sol: np.ndarray, problem's solution 
    :param method: string, the available ERGM methods:
        'dcm':
        'dcm_rd':
    :param d: ordered Dict, contains the info about the reduced system
    :return:
    """
    if method == 'dcm':
        if t == 'dyads':
            return std_dyads_dcm(sol)
        if t == 'singles':
            return std_singles_dcm(sol)
        if t == 'zeros':
            return std_zeros_dcm(sol)

    if method == 'dcm_rd':
        d = scalability_classes(A, 'dcm_rd')
        sol_full = rd2full(sol, d, 'dcm_rd')
 
        if t == 'dyads':
            return std_dyads_dcm(sol_full)
        if t == 'singles':
            return std_singles_dcm(sol_full)
        if t == 'zeros':
            return std_zeros_dcm(sol_full)



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
            temp += y[j]*x[i]/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= x[i]*y[i]/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
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
def std_dyads_dcm(sol):
    """ compute the expected number of reciprocated links 
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j]) 
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += 2*pij*pji*(1 - pij*pji) 
    return np.sqrt(temp)


@jit(nopython=True)
def std_singles_dcm(sol):
    """ compute the expected number of reciprocated links 
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j]) 
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += pij*(1 - pji)*(1 - pij*(1 - pji) - pji*(1 - pij))  
    return np.sqrt(temp)


@jit(nopython=True)
def std_zeros_dcm(sol):
    """ compute the expected number of zeros couples 
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j]) 
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += 2*(1 - pij)*(1 - pji)*(1 - (1 - pij)*(1 - pji)) 
    return np.sqrt(temp)


@jit(nopython=True)
def expected_dyads_dcm_rd(sol, c):
    #TODO: idoesn't work
    n = int(len(sol)/2)
    y = sol[:n] #TODO: tmeporary fix from an old notation
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


def ensemble_sampler(sol, m, method, sample_dir='.', start=0, seed=None):
    """ samples m adjacency matrices in diretory sampler_dir, after the method ergm solution given by sol
    """
    np.random.seed(seed)
    # if sample_dir doesn't exists, creates it
    try:
        os. mkdir(sample_dir)
    except FileExistsError:
        pass

    if method == 'cm':
        n = len(sol)
        x = sol
        for k in range(start, start + m):
            r = np.random.random((n, n))
            a = np.outer(x, x)/(np.ones((n, n)) + np.outer(x, x))
            np.fill_diagonal(a, 0)
            c = np.zeros((n, n))
            c[a.__gt__(r)] = 1
            del r, a
            sparse_matrix = scipy.sparse.coo_matrix(c)
            del c

            outfile = sample_dir + '/' + 'cm_graph_{}.npz'.format(k)
            scipy.sparse.save_npz(outfile, sparse_matrix)

    if method == 'dcm':
        n = int(len(sol)/2)
        x = sol[:n]
        y = sol[n:]
        # sampling
        for k in range(start, start + m):
            r = np.random.random((n, n))
            a = np.outer(x, y)/(np.ones((n, n)) + np.outer(x, y))
            np.fill_diagonal(a, 0)
            c = np.zeros((n, n))
            c[a.__gt__(r)] = 1
            del r, a
            sparse_matrix = scipy.sparse.coo_matrix(c)
            del c

            outfile = sample_dir + '/' + 'dcm_graph_{}.npz'.format(k)
            scipy.sparse.save_npz(outfile, sparse_matrix)

        return 0


def probability_matrix(sol, method):
    if method == 'dcm':
        n = int(len(sol)/2)
        x = sol[:n]
        y = sol[n:]
        p = np.outer(x, y)/(np.ones((n, n)) + np.outer(x, y))
        np.fill_diagonal(p, 0)
        return p
