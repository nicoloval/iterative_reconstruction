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
    
    def test_dcm_dyads(self):
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
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm') 

        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)


    def test_dcm_singles(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        singles_count = 1 #TODO: calculate! 
        dyads_analytical_expectation = 10 
        sol = np.array([1]*10)
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm', t='singles') 
        #print(dyads_fun_expectation)

        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)

    def test_dcm_zeros(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        zeros_count = 1 #TODO: calculate! 
        dyads_analytical_expectation = 5 
        sol = np.array([1]*10)
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm', t='zeros') 
        print(dyads_fun_expectation)

        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)



    def test_dcm_rd_dyads(self):
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
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm_rd', A=A) 
        #print(dyads_fun_expectation)
        #print(dyads_analytical_expectation)
        self.assertTrue(dyads_fun_expectation == dyads_analytical_expectation)
        
 
if __name__ == '__main__':
    unittest.main()

