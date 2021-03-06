'''Test function iterative_solver 
'''
import sys
sys.path.append('../')
from sample import iterative_solver # where the functions are
import sample
import numpy as np
from numba import jit
import unittest  # test tool
import networkx as nx
import scipy.sparse


class MyTest(unittest.TestCase):

    def setUp(self):
        pass
    

    def test_array_sym(self):
        A = np.array([[0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0]])
        
        A_nx = nx.convert_matrix.from_numpy_array(A)
        k = sample.out_degree(A)

        knn_sample = sample.nearest_neighbour_degree_undirected(A)
        # knn_nx = nx.degree_assortativity_coefficient(A_nx)
        knn_nx = nx.k_nearest_neighbors(A_nx)

        # debug check
        """
        print(k)
        print('\n')
        print(knn_sample)
        print(knn_nx)
        """

        self.assertTrue(knn_sample == knn_nx)


    def test_sparse_sym(self):
        A = np.array([[0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0]])
        
        A_nx = nx.convert_matrix.from_numpy_array(A)

        # sparse matrix conversion
        A = scipy.sparse.csr_matrix(A)

        rows, cols = A.nonzero()
        print(rows, cols)
        A[cols,rows] = A[rows, cols]


        k = sample.out_degree(A)

        knn_sample = sample.nearest_neighbour_degree_undirected(A)
        # knn_nx = nx.degree_assortativity_coefficient(A_nx)
        knn_nx = nx.k_nearest_neighbors(A_nx)

        # debug check
        """
        print(k)
        print('\n')
        print(knn_sample)
        print(knn_nx)
        """

        self.assertTrue(knn_sample == knn_nx)




    def test_array_inin(self):
        A = np.array([[0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0]])
        
        A_nx = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph())
        k = sample.in_degree(A)

        knn_sample = sample.nearest_neighbour_degree_inin(A)
        # knn_nx = nx.degree_assortativity_coefficient(A_nx)
        knn_nx = nx.k_nearest_neighbors(A_nx, source='in', target='in')

        # debug check
        """
        print(k)
        print('\n')
        print(knn_sample)
        print(knn_nx)
        """

        self.assertTrue(knn_sample == knn_nx)


    def test_array_outout(self):
        A = np.array([[0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0]])
        
        A_nx = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph())
        k = sample.out_degree(A)

        knn_sample = sample.nearest_neighbour_degree_outout(A)
        # knn_nx = nx.degree_assortativity_coefficient(A_nx)
        knn_nx = nx.k_nearest_neighbors(A_nx, source='out', target='out')

        # debug check
        """
        print(k)
        print('\n')
        print(knn_sample)
        print(knn_nx)
        """

        self.assertTrue(knn_sample == knn_nx)


if __name__ == '__main__':
    unittest.main()

