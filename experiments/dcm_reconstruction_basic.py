import sys
sys.path.append('../')
import sample
import time
import numpy as np
class bcolors:
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'

def tester(A, verbose=False):
        t = time.time()
        tic = time.time()
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        k = np.concatenate((k_out, k_in))
        toc = time.time() - tic
        if len(k) < 12:
                print("\n\nExperiment matrix: \n {}".format(A))
                print("\n degree sequence:\n{}".format(k))
        print('\ndegree sequence computation time = {}'.format(toc))
        # dianati iterative function for dcm
        tic = time.time()
        n = len(k_in)
        x0 = np.array([ki/np.sqrt(n) for ki in k])  # initialial point
        sol, step, diff = sample.iterative_solver(A, max_steps = 300, eps = 0.01, method='dcm')
        toc = time.time() - tic
        print('\nsolver exectution time = {}'.format(toc))
        # test results: degree reconstruction
        tic = time.time()
        k_in_sol = sample.expected_in_degree_dcm(sol)
        k_out_sol = sample.expected_out_degree_dcm(sol)
        k_sol = np.concatenate((k_out_sol, k_in_sol))
        toc = time.time() - tic
        print('\n number of steps = {}'.format(step))
        print('\nsolution = {}'.format(sol))
        print('\ndegree reconstruction time = {}'.format(toc))
        print('\n{}reconstruction error = {}{}'.format(bcolors.WARNING, np.linalg.norm(k - k_sol), bcolors.ENDC))
        if len(k) < 12:
                print("\n Original degree sequence:\n{}".format(k))
                print("\n Reconstructed degree sequence\n{}".format(k_sol))
        toc = time.time() - t
        print('\ntotal test time = {}'.format(toc))

# test matrix 3
A = np.array([[0, 1, 1, 1],
              [1, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
tester(A)  # works

