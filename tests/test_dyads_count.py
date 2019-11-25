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
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        
        actual_dyads = 0
        actual_singles = 5
        actual_zeros = 1

        dyads = sample.dyads_count(A)
        singles = sample.singles_count(A)
        zeros = sample.zeros_count(A)

        # debug check
        # print(dyads)
        # print(singles)
        # print(zeros)
        self.assertTrue(np.allclose(dyads, actual_dyads, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(singles, actual_singles, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(zeros, actual_zeros, atol=1e-02, rtol=1e-02))


    def test_sparse_dcm(self):
        A = np.array([[0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0]])
        sA = scipy.sparse.csr_matrix(A)
        
        actual_dyads = 0
        actual_singles = 5
        actual_zeros = 1

        dyads = sample.dyads_count(sA)
        singles = sample.singles_count(sA)
        zeros = sample.zeros_count(sA)

        self.assertTrue(np.allclose(dyads, actual_dyads, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(singles, actual_singles, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(zeros, actual_zeros, atol=1e-02, rtol=1e-02))


if __name__ == '__main__':
    unittest.main()

