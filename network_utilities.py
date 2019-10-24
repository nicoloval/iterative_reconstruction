import numpy as np
import scipy.sparse
from numba import jit

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


def expected_out_degree(sol, method, par=None):
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
        return expected_out_degree_dcm_rd(sol, par)


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
def expected_out_degree_dcm_rd(sol, par):
    n = int(sol.size/2)
    a_out = sol[0:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    c = par[2*n:3*n]  # cardinality of reduced equivalent classes

    for i in range(n):
        for j in range(n):
            if j != i:
                k[i] += c[j]*a_in[j]*a_out[i]/(1 + a_in[j]*a_out[i])
            else:
                k[i] += (c[i] - 1)*a_in[i]*a_out[i]/(1 + a_in[i]*a_out[i])
    return k


def expected_in_degree(sol, method, par=None):
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
        return expected_in_degree_dcm_rd(sol, par)


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
def expected_in_degree_dcm_rd(sol, par):
    n = int(sol.size/2)
    a_out = sol[0:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    c = par[2*n:3*n]  # cardinality of reduced equivalent classes

    for i in range(n):
        for j in range(n):
            if j != i:
                k[i] += c[j]*a_out[j]*a_in[i]/(1 + a_out[j]*a_in[i])
            else:
                k[i] += (c[i] - 1)*a_out[i]*a_in[i]/(1 + a_out[i]*a_in[i])

    return k


