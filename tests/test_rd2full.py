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
    
    def test_dcm_rd(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        right_d = {
                (2, 2):[0, 1],
                (1, 3):[2],
                (3, 1):[3],
                (1, 1):[4]
            } 

        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        k = np.concatenate((k_out, k_in))

        d = sample.scalability_classes(A, 'dcm_rd')

        self.assertTrue(np.alltrue(sample.rd2full_dcm_rd(np.array([2, 1, 3, 1, 2, 3, 1, 1]), d) == k))

        
 
if __name__ == '__main__':
    unittest.main()

