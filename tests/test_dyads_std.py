'''Test function dyads_expectation calculation
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
        dyads_analytical_std = np.sqrt(7.5) 
        sol = np.array([1]*10)
        dyads_fun_std = sample.std_dyads(sol, method='dcm') 

        self.assertTrue(dyads_fun_std == dyads_analytical_std)


    def test_dcm_singles(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        singles_count = 6 
        dyads_analytical_std = np.sqrt(2.5) 
        sol = np.array([1]*10)
        dyads_fun_std = sample.std_dyads(sol, method='dcm', t='singles') 
        # debug check
        # print(dyads_analytical_std)
        # print(dyads_fun_std)

        self.assertTrue(dyads_fun_std == dyads_analytical_std)

    def test_dcm_zeros(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        zeros_count = 2 #TODO: calculate! 
        dyads_analytical_std = np.sqrt(7.5) 
        sol = np.array([1]*10)
        dyads_fun_std = sample.std_dyads(sol, method='dcm', t='zeros') 
        # print(dyads_fun_expectation)

        self.assertTrue(dyads_fun_std == dyads_analytical_std)


    #TODO: the rd methods are just a variant of full methods rn
 
if __name__ == '__main__':
    unittest.main()

