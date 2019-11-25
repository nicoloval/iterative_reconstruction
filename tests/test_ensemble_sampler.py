'''Test function iterative_solver 
'''
import sys
sys.path.append('../')
from sample import iterative_solver # where the functions are
import sample
import numpy as np
from numba import jit
import unittest  # test tool
import scipy.sparse


class MyTest(unittest.TestCase):

    def setUp(self):
        pass
    

    def test_sparse_dcm(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        sA = scipy.sparse.csr_matrix(A)
        # debug check
        # sk_out = sample.out_degree(sA)
        # sk_in = sample.in_degree(sA)
        # sk = np.concatenate((sk_out, sk_in))
        ##  print(sk) 
        sol, step, diff = iterative_solver(sA, max_steps = 300, eps = 0.01)
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_out_it = sample.expected_out_degree(sol, 'dcm')
        expected_in_it = sample.expected_in_degree(sol, 'dcm')
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        sample.ensemble_sampler(sol=sol, m=1, method='dcm')
        AA = scipy.sparse.load_npz('dcm_graph_0.npz')
        # debug check
        k_out = sample.out_degree(AA)
        k_in = sample.in_degree(AA)
        k = np.concatenate((k_out, k_in))
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))


if __name__ == '__main__':
    unittest.main()

