'''Run the iterative network reconstruiction solver on some bitcoin networks

* 2011-01-17.graphml: N = 2k E= 4K
* 2012-01-16.graphml
'''
import sys
sys.path.append('../')
import sample # where the functions are
import network_utilities as utils
import numpy as np
import scipy.sparse
import networkx as nx
import time

# btc database absolute path
# db_path = '~/Datasets/NET-btc530k-heur_2s-week/'
db_path = '/mnt/hdd_data/imt/NET-btc530k-heur_2s-week/'
# week to test
# week = '2011-01-17.graphml'  
# week = '2012-01-16.graphml'  
# week = '2013-01-14.graphml'  
week = '2014-01-13.graphml'  
# load the graph
G = nx.read_graphml(db_path + week)
# A = nx.to_numpy_array(G, dtype=int)
A = scipy.sparse(G) 
# iterative solver
tic = time.time()
sol, step, diff = sample.iterative_solver(A, eps=1e-04)
t = time.time() - tic
# test solution
expected_out_it = utils.expected_out_degree(sol)
expected_in_it = utils.expected_in_degree(sol)
expected_k = np.concatenate((expected_out_it, expected_in_it))
k_out = utils.out_degree(A)
k_in = utils.in_degree(A)
k = np.concatenate((k_out, k_in))
#
result = np.allclose(expected_k, k, atol=1e-02, rtol=1e-02)
print('experiment success: {}'.format(result))
if not result:
    print('error = {}'.format(max(abs(k - expected_k))))
#
print('diff = {}'.format(diff) )
print('steps = {}'.format(step) )
print('time = {}'.format(t) )
