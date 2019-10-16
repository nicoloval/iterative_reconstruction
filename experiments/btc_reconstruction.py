'''Run the iterative network reconstruiction solver on some bitcoin networks

* 2011-01-17.graphml: N = 2k E= 4K
'''
import sys
sys.path.append('../')
from sample import iterative_fun # where the functions are
import numpy as np
import networkx as nx

# btc database absolute path
db_path = '~/Datasets/NET-btc530k-heur_2s-week/'
# week to test
week = '2011-01-17.graphml'  
# load the graph
G = nx.read_graphmk(db_path + week)


