'''Run the iterative network reconstruiction solver on some bitcoin networks
using the dcm_rd method

each time is run, saves results on a csv.
This script should be run on consensus server
'''
import sys
import os
sys.path.append('../')
import sample # where the functions are
import numpy as np
import scipy.sparse
import networkx as nx
import time
import datetime

# btc database absolute path
db_path = '/mnt/hdd_data/imt/NET-btc530k-heur_2s-week/'
# week to test
# week = '2011-01-17.graphml'  
# week = '2012-01-16.graphml'  
# week = '2013-01-14.graphml'  
# week = '2014-01-13.graphml'  
week = sys.argv[1]

print(db_path + week)
# load the graph
G = nx.read_graphml(db_path + week)
# A = nx.to_numpy_array(G, dtype=int)
A = nx.to_scipy_sparse_matrix(G) 

# iterative solver
tic = time.time()
sol, step, diff = sample.iterative_solver(A, eps=1e-04, method='dcm_rd')
t = time.time() - tic

# test solution
d = sample.scalability_classes(A, 'dcm_rd')
expected_out_it = sample.expected_out_degree(sol, 'dcm_rd', d)
expected_in_it = sample.expected_in_degree(sol, 'dcm_rd', d)
expected_k = np.concatenate((expected_out_it, expected_in_it))
k_out = sample.out_degree(A)
k_in = sample.in_degree(A)
k = np.concatenate((k_out, k_in))
error = max(abs(k - expected_k))
result = np.allclose(expected_k, k, atol=1e-02, rtol=1e-02)

# output
print('experiment success: {}'.format(result))
if not result:
    print('error = {}'.format(error))
print('diff = {}'.format(diff) )
print('steps = {}'.format(step) )
print('time = {}'.format(t) )

# save results
csvfile = 'df_btc_dcmrd.csv'
# if the csv file already exists, do not write the header
head_line = 'experiment_date,file,succes,error,diff,steps,time\n'
if not os.path.isfile(csvfile):
    with open(csvfile, 'w') as cfile:
        cfile.write(head_line)

date = datetime.date.today()
# writing the df
line = '{},{},{},{},{},{},{}\n'.format(date, week, result, error, diff, step, t )
with open(csvfile, 'a') as cfile:
    cfile.write(line)
