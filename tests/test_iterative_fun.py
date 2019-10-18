'''Test function iterative_fun which calculate the iterative step 
'''
import sys
sys.path.append('../')
import sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_ones_dcm(self):
        k_out = np.array([2, 2, 0, 1])
        k_in = np.array([0, 1, 3, 1])
        par = np.concatenate((k_out, k_in))

        x = np.array([1, 1, 1, 1])
        y = np.array([1, 1, 1, 1])
        v = np.concatenate((x, y))

        right_x = np.array([4/3, 4/3, 0, 2/3])
        right_y = np.array([0, 2/3, 2, 2/3])
        right_v = np.concatenate((right_x, right_y))

        self.assertTrue(np.alltrue(sample.iterative_fun_dcm(v, par) == right_v))
        
        # TODO: implementare un test che prendi x e y come i punti iniziali

if __name__ == '__main__':
    unittest.main()

