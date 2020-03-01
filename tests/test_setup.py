'''Test function setup 
'''
import sys
sys.path.append('../')
import sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass
    

    def test_cm(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0]]
        )

        l = sample.setup(A, method='cm')

        right_par = np.array([2, 2, 3, 2, 1])
        right_v0 = right_par/np.sqrt(A.sum())

        self.assertTrue(np.alltrue(l[0] == right_par) & np.alltrue(l[1] == right_v0))

 
    def test_dcm(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )

        l = sample.setup(A, method='dcm')

        right_par = np.array([2, 2, 1, 3, 1, 2, 2, 3, 1, 1])
        right_v0 = right_par/np.sqrt(A.sum())

        self.assertTrue(np.alltrue(l[0] == right_par) & np.alltrue(l[1] == right_v0))
        

    def test_dcm_rd(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )

        l = sample.setup(A, method='dcm_rd')

        right_par = np.array([2, 1, 3, 1, 2, 3, 1, 1, 2, 1, 1, 1])
        right_v0 = np.array([2, 1, 3, 1, 2, 3, 1, 1])/np.sqrt(A.sum())

        self.assertTrue(np.alltrue(l[0] == right_par) & np.alltrue(l[1] == right_v0))


    def test_decm(self):
        A = np.array(
                [[0, 1, 2, 0],
                [2, 0, 3, 0],
                [0, 0, 0, 3],
                [3, 1, 0, 0]]
        )

        l = sample.setup(A, method='decm')

        right_par = np.array( 
            [2, 2, 1, 2, 2, 2, 2, 1, 3, 5, 3, 4, 5, 2, 5, 3]
            )
        right_v0 = right_par/np.concatenate(
            (np.sqrt(7)*np.ones(8), np.sqrt(15)*np.ones(8))
            )

        self.assertTrue(np.alltrue(l[0] == right_par) & np.alltrue(l[1] == right_v0))
 
 
if __name__ == '__main__':
    unittest.main()

