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
    

    def test_array_dcm(self):
        A = np.array([[0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0]])
        
        actual_k_out_nr = np.zeros(5)
        actual_k_in_nr = np.zeros(5)
        actual_k_r = np.zeros(5)
        n = len(A)
        for i in range(n):
            for j in range(n):
                if i != j:
                    actual_k_r[i] += A[i,j]*A[j,i]
                    actual_k_out_nr[i] += A[i,j]*(1 - A[j,i])
                    actual_k_in_nr[i] += A[j,i]*(1 - A[i,j])

        k_out_nr = sample.non_reciprocated_out_degree(A)
        k_in_nr = sample.non_reciprocated_in_degree(A)
        k_r = sample.reciprocated_degree(A)

        """
        # debug check
        print('\n')
        print(actual_k_r)
        print(k_r)
        print(actual_k_out_nr)
        print(k_out_nr)
        print(actual_k_in_nr)
        print(k_in_nr)
        """

        self.assertTrue(np.allclose(k_out_nr, actual_k_out_nr, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(k_in_nr, actual_k_in_nr, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(k_r, actual_k_r, atol=1e-02, rtol=1e-02))



    def test_sparse_dcm(self):
        A = np.array([[0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0]])

        A = scipy.sparse.csr_matrix(A) 
        
        actual_k_out_nr = np.zeros(5)
        actual_k_in_nr = np.zeros(5)
        actual_k_r = np.zeros(5)
        n = 5
        for i in range(n):
            for j in range(n):
                if i != j:
                    actual_k_r[i] += A[i,j]*A[j,i]
                    actual_k_out_nr[i] += A[i,j]*(1 - A[j,i])
                    actual_k_in_nr[i] += A[j,i]*(1 - A[i,j])

        k_out_nr = sample.non_reciprocated_out_degree(A)
        k_in_nr = sample.non_reciprocated_in_degree(A)
        k_r = sample.reciprocated_degree(A)

        """
        # debug check
        print('\n')
        print(actual_k_r)
        print(k_r)
        print(actual_k_out_nr)
        print(k_out_nr)
        print(actual_k_in_nr)
        print(k_in_nr)
        """

        self.assertTrue(np.allclose(k_out_nr, actual_k_out_nr, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(k_in_nr, actual_k_in_nr, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(k_r, actual_k_r, atol=1e-02, rtol=1e-02))


if __name__ == '__main__':
    unittest.main()

