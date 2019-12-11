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

        sol = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        actual_p = np.array([[0, 0.5, 0.5, 0.5],
            [0.5, 0, 0.5, 0.5],
            [0.5, 0.5, 0, 0.5],
            [0.5, 0.5, 0.5, 0]]
            )

        p = sample.probability_matrix(sol, method='dcm')

        self.assertTrue(np.allclose(p, actual_p, atol=1e-02, rtol=1e-02))


if __name__ == '__main__':
    unittest.main()

