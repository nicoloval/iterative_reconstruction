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
    

    def test_ones_cm(self):
        par = np.array([1, 3, 2, 2])

        v = np.array([1, 1, 1, 1])

        right_v = np.array([2/3, 2, 4/3, 4/3])

        # debug
        # print(right_v)
        # print(sample.iterative_fun_cm(v, par)) 
        self.assertTrue(np.alltrue(sample.iterative_fun_cm(v, par) == right_v))


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


    def test_ones_dcm_rd(self):
        k_out = np.array([0, 1, 2])
        k_in = np.array([2, 2, 0])
        c = np.array([1, 2, 2])
        par = np.concatenate((k_out, k_in, c))

        x = np.array([1, 1, 1])
        y = np.array([1, 1, 1])
        v = np.concatenate((x, y))

        right_x = np.array([0, 0.5, 1])
        right_y = np.array([1, 1, 0])
        right_v = np.concatenate((right_x, right_y))

        self.assertTrue(np.alltrue(sample.iterative_fun_dcm_rd(v, par) == right_v))


    def test_ones_rdcm(self):
        A = np.array([[0, 1, 1, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 1],
                     [0, 1, 0, 0]])
        k_out_nr = np.array([2, 0, 2, 1])
        k_in_nr = np.array([0, 2, 1, 2])
        k_r = np.array([1, 1, 0, 0])
        par = np.concatenate((k_out_nr, k_in_nr, k_r))

        x = np.array([1, 1, 1, 1])
        y = np.array([1, 1, 1, 1])
        z = np.array([1, 1, 1, 1])

        v = np.concatenate((x, y, z))

        right_x = np.array([8/3, 0, 8/3, 4/3])
        right_y = np.array([0, 8/3, 4/3, 8/3])
        right_z = np.array([4/3, 4/3, 0, 0])
        right_v = np.concatenate((right_x, right_y, right_z))

        self.assertTrue(np.alltrue(sample.iterative_fun_rdcm(v, par) == right_v))
 
        
 
if __name__ == '__main__':
    unittest.main()

