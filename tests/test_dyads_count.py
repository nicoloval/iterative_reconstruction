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
    

    def test_array_dcm_2(self):
        A = np.array([[0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0]])
        
        actual_dyads = 0
        actual_singles = 0
        actual_zeros = 0
        n = len(A)
        for i in range(n):
            for j in range(n):
                if i != j:
                    actual_dyads += A[i,j]*A[j,i]
                    actual_singles += A[i,j]*(1 - A[j,i])
                    actual_zeros += (1 - A[i,j])*(1 - A[j,i])

        dyads = sample.dyads_count(A)
        singles = sample.singles_count(A)
        zeros = sample.zeros_count(A)

        # debug check
        """
        print('\n')
        print(actual_dyads)
        print(actual_singles)
        print(actual_zeros)
        print(dyads)
        print(singles)
        print(zeros)
        """
        self.assertTrue(np.allclose(dyads, actual_dyads, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(singles, actual_singles, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(zeros, actual_zeros, atol=1e-02, rtol=1e-02))



if __name__ == '__main__':
    unittest.main()

