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


    def test_array_cm(self):
        A = np.array([[0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 0],
                [0, 1, 0, 0]])
        
        sol, step, diff = iterative_solver(A, max_steps = 300, eps = 0.01, method='cm')
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_k = sample.expected_degree(sol, 'cm')
        k = sample.out_degree(A)
        # debug check
        print(k)
        print(expected_k)
        print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))

    
    def test_array_dcm(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        
        sol, step, diff = iterative_solver(A, max_steps = 300, eps = 0.01)
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_out_it = sample.expected_out_degree(sol, 'dcm')
        expected_in_it = sample.expected_in_degree(sol, 'dcm')
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))


    def test_sparse_dcm(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        sA = scipy.sparse.csr_matrix(A)
        
        sol, step, diff = iterative_solver(sA, max_steps = 300, eps = 0.01)
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        expected_out_it = sample.expected_out_degree(sol, 'dcm')
        expected_in_it = sample.expected_in_degree(sol, 'dcm')
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = sample.out_degree(sA)
        k_in = sample.in_degree(sA)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))


    def test_array_dcm_rd(self):
        A = np.array([[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]])
  
        sol, step, diff = iterative_solver(A, max_steps = 300, eps = 0.01, method='dcm_rd')
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        d = sample.scalability_classes(A, method='dcm_rd')
        expected_out_it = sample.expected_out_degree(sol, 'dcm_rd', d)
        expected_in_it = sample.expected_in_degree(sol, 'dcm_rd', d)
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(d)
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))


        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))


    def test_sparse_dcm_rd(self):
        A = np.array([[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]])

        sA = scipy.sparse.csr_matrix(A)
        
        sol, step, diff = iterative_solver(sA, max_steps = 300, eps = 0.01, method='dcm_rd')
        # output convergence 
        # print('steps = {}'.format(step))
        # print('diff = {}'.format(diff))
        # epectation degree vs actual degree
        d = sample.scalability_classes(A, method='dcm_rd')
        expected_out_it = sample.expected_out_degree(sol, 'dcm_rd', d)
        expected_in_it = sample.expected_in_degree(sol, 'dcm_rd', d)
        expected_k = np.concatenate((expected_out_it, expected_in_it))
        k_out = sample.out_degree(sA)
        k_in = sample.in_degree(sA)
        k = np.concatenate((k_out, k_in))
        # debug check
        # print(d)
        # print(k)
        # print(expected_k)
        # print(np.linalg.norm(k- expected_k))

        self.assertTrue(np.allclose(expected_k, k, atol=1e-02, rtol=1e-02))



if __name__ == '__main__':
    unittest.main()

