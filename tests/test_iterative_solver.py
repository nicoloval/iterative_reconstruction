'''Test function iterative_solver 
'''
import sys
sys.path.append('../')
from sample import iterative_solver # where the functions are
from network_utilities import *
import numpy as np
from numba import jit
import unittest  # test tool
import scipy.sparse


class MyTest(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_array(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        
        sol, step, diff = iterative_solver(A, max_steps = 300, eps = 0.01)
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_out_it = expected_out_degree(sol)
        expected_in_it = expected_in_degree(sol)
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = out_degree(A)
        k_in = in_degree(A)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))

    def test_sparse(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        sA = scipy.sparse.csr_matrix(A)
        
        sol, step, diff = iterative_solver(sA, max_steps = 300, eps = 0.01)
        # output convergence 
        print('steps = {}'.format(step))
        print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_out_it = expected_out_degree(sol)
        expected_in_it = expected_in_degree(sol)
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = out_degree(sA)
        k_in = in_degree(sA)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))
        

if __name__ == '__main__':
    unittest.main()

