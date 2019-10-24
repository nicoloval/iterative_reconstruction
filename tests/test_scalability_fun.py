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
    
    def test_1(self):
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

        self.assertTrue(sample.scalability_classes(A, method='dcm_rd')  == right_d)
        
 
if __name__ == '__main__':
    unittest.main()

