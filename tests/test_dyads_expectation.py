'''Test function dyads_expectation calculation
'''
import os
import sys
sys.path.append('../')
import sample
import numpy as np
import scipy.sparse
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
        
        dyads_analytical_expectation = 5 
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
        
        dyads_analytical_expectation = 5 
        sol = np.array([1]*10)
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm', t='zeros') 
        # print(dyads_fun_expectation)

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
        

    def test_dcm_dyads_emseble_empirical_1(self):
        A = np.array(
                [[0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0]]
        )
        
        # solve the netrec problem
        sol, step, diff = sample.iterative_solver(A, max_steps = 300, eps = 0.01)
        # find the analytical expectation
        # dyads
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm') 
        # singles
        singles_fun_expectation = sample.expected_dyads(sol, method='dcm', t='singles') 
        # zeros
        zeros_fun_expectation = sample.expected_dyads(sol, method='dcm', t='zeros') 
        # sample 100 networks (dcm method by default)
        s_dir = 'tmp'
        sample.ensemble_sampler(sol=sol, m=100, method='dcm', sample_dir=s_dir)
        # count empirical dyads for each sampled network 
        files = os.listdir(s_dir)
        dyads_l = []
        singles_l = []
        zeros_l = []
        for f in files:
            fA = scipy.sparse.load_npz(s_dir + '/' + f)
            dyads = sample.dyads_count(fA)
            dyads_l.append(dyads)
            singles = sample.singles_count(fA)
            singles_l.append(singles)
            zeros = sample.zeros_count(fA)
            zeros_l.append(zeros)
        # compute empirical average
        dyads_empirical_expectation = np.average(dyads_l) 
        singles_empirical_expectation = np.average(singles_l) 
        zeros_empirical_expectation = np.average(zeros_l) 
        # debug
        print('Empirical test 1')
        print('dyads')
        print(dyads_fun_expectation)
        print(dyads_empirical_expectation)
        print('singles')
        print(singles_fun_expectation)
        print(singles_empirical_expectation)
        print('zeros')
        print(zeros_fun_expectation)
        print(zeros_empirical_expectation)
        # remove ensemble directory
        files = os.listdir(s_dir)
        for f in files:
            os.remove(s_dir + '/' + f)
        os.rmdir(s_dir)
        # testing
        self.assertTrue(np.allclose(dyads_fun_expectation, dyads_empirical_expectation, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(singles_fun_expectation, singles_empirical_expectation, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(zeros_fun_expectation, zeros_empirical_expectation, atol=1e-02, rtol=1e-02))


 
    def test_dcm_dyads_emseble_empirical_2(self):
        A = np.array([[0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0]]
            )

        
        # solve the netrec problem
        sol, step, diff = sample.iterative_solver(A, max_steps = 300, eps = 0.01)
        # find the analytical expectation
        # dyads
        dyads_fun_expectation = sample.expected_dyads(sol, method='dcm') 
        # singles
        singles_fun_expectation = sample.expected_dyads(sol, method='dcm', t='singles') 
        # zeros
        zeros_fun_expectation = sample.expected_dyads(sol, method='dcm', t='zeros') 
        # sample 100 networks (dcm method by default)
        s_dir = 'tmp'
        sample.ensemble_sampler(sol=sol, m=100, method='dcm', sample_dir=s_dir)
        # count empirical dyads for each sampled network 
        files = os.listdir(s_dir)
        dyads_l = []
        singles_l = []
        zeros_l = []
        for f in files:
            fA = scipy.sparse.load_npz(s_dir + '/' + f)
            dyads = sample.dyads_count(fA)
            dyads_l.append(dyads)
            singles = sample.singles_count(fA)
            singles_l.append(singles)
            zeros = sample.zeros_count(fA)
            zeros_l.append(zeros)
        # compute empirical average
        dyads_empirical_expectation = np.average(dyads_l) 
        singles_empirical_expectation = np.average(singles_l) 
        zeros_empirical_expectation = np.average(zeros_l) 
        # debug
        print('Empirical test 2')
        print('dyads')
        print('analytical: {}'.format(dyads_fun_expectation))
        print('Empirical : {}'.format(dyads_empirical_expectation))
        print('singles')
        print(singles_fun_expectation)
        print(singles_empirical_expectation)
        print('zeros')
        print(zeros_fun_expectation)
        print(zeros_empirical_expectation)
        # remove ensemble directory
        files = os.listdir(s_dir)
        for f in files:
            os.remove(s_dir + '/' + f)
        os.rmdir(s_dir)
        # testing
        self.assertTrue(np.allclose(dyads_fun_expectation, dyads_empirical_expectation, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(singles_fun_expectation, singles_empirical_expectation, atol=1e-02, rtol=1e-02))
        self.assertTrue(np.allclose(zeros_fun_expectation, zeros_empirical_expectation, atol=1e-02, rtol=1e-02))

if __name__ == '__main__':
    unittest.main()

