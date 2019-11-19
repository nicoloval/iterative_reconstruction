'''Test function scalability_classes
'''
import sys
sys.path.append('../')
import sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_dcm(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        dyads_count = 1 
        dyads_analytical_expectation = 5 
        sol = np.array([1]*10)
        dyads_fun_expectation = sample.expected_dyads_dcm(sol) 

        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)

    def test_dcm_rd(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        dyads_count = 1 
        dyads_analytical_expectation = 5 
        sol = np.array([1]*8)
        """
        c = [2, 0, 0, 0]
        dyads_fun_expectation = sample.expected_dyads_dcm_rd(sol, c) 
        print(dyads_fun_expectation)
        print(dyads_analytical_expectation)
        """
        d = sample.scalability_classes(A, 'dcm_rd')
        sol_full = sample.rd2full(sol, d, 'dcm_rd')
        dyads_fun_expectation = sample.expected_dyads_dcm(sol) 
        print(dyads_fun_expectation)
        print(dyads_analytical_expectation)
        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)
        
 
if __name__ == '__main__':
    unittest.main()

