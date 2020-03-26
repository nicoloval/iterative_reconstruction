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
        s_out = sample.out_strength(A)
        s_in = sample.in_strength(A)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        toc = time.time() - tic
        if len(k) < 30:
                print("\n\nExperiment matrix: \n {}".format(A))
                print("\n degree sequence:\n{}".format(k))
        print('\ndegree sequence computation time = {}'.format(toc))
        # dianati iterative function for dcm
        tic = time.time()
        n = len(k_in)
        par, x0 = sample.setup(A, method='decm')
        print('initial guess = {}'.format(x0))
        sol, step, diff = sample.iterative_solver(A, max_steps = 300, eps = 1e-4, method='decm', verbose=verbose)
        toc = time.time() - tic
        print('\nsolver exectution time = {}'.format(toc))
        # test results: degree reconstruction
        tic = time.time()
        k_in_sol = sample.expected_in_degree_decm(sol)
        k_out_sol = sample.expected_out_degree_decm(sol)
        s_in_sol = sample.expected_in_strength_decm(sol)
        s_out_sol = sample.expected_out_strength_decm(sol)
        k_sol = np.concatenate((k_out_sol, k_in_sol, s_out_sol, s_in_sol))
        toc = time.time() - tic
        print('\n number of steps = {}'.format(step))
        print('\nsolution = {}'.format(sol))
        print('\ndegree reconstruction time = {}'.format(toc))

        print('\n{}reconstruction error = {}{}'.format(bcolors.WARNING, np.linalg.norm(k - k_sol), bcolors.ENDC))
        if len(k) < 30:
                print("\n Original degree sequence:\n{}".format(k))
                print("\n Reconstructed degree sequence\n{}".format(k_sol))
        toc = time.time() - t
        print('\ntotal test time = {}'.format(toc))


# test matrix 1
A = np.array([[0, 2, 4, 1],
              [1, 0, 1, 0],
              [0, 0, 0, 3],
              [1, 2, 0, 0]])
tester(A, verbose=True)  # works
"""
# test matrix 2
A = np.array([[0, 3, 2, 1],
              [1, 0, 2, 0],
              [0, 0, 0, 5],
              [0, 0, 0, 0]])
tester(A)  # works

# test matrix 3
A = np.array([[0, 2, 3, 0],
              [1, 0, 3, 0],
              [0, 1, 0, 0],
              [3, 0, 5, 0]])
tester(A)  # works
"""
