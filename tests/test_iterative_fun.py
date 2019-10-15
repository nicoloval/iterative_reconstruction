'''Test function iterative_fun which calculate the iterative step 
'''
import sys
sys.path.append('../')
from sample import iterative_fun # where the functions are
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def check_truth(self, t, t1):
        # unpack t and t1
        x, y = t
        x1, y2 = t1
        a = np.concatenate((x, y))
        b = np.concatenate((x1, y2))

        return np.alltrue(a == b)

    def test_ones(self):
        k_out = np.array([2, 2, 0, 1])
        k_in = np.array([0, 1, 3, 1])
        x = np.array([1, 1, 1, 1])
        y = np.array([1, 1, 1, 1])
        right_x = np.array([4/3, 4/3, 0, 2/3])
        right_y = np.array([0, 2/3, 2, 2/3])
        self.assertTrue(self.check_truth(iterative_fun(x, y, k_out, k_in), (right_x, right_y)))
        
        # TODO: implementare un test che prendi x e y come i punti iniziali

if __name__ == '__main__':
    unittest.main()

