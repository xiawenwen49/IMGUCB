
import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import operator
import os
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager

def generate_subnode_list(save_dir, edge_file):
	NodeDegree = {}

	with open(edge_file) as f:
		counter = 0
		for line in f:
			if counter >=4:
				data = line.split()
				u = int(data[0])
				v = int(data[1])
				if u not in NodeDegree: 
					NodeDegree[u] = 1
				else:
					NodeDegree[u]  +=1
				if v not in NodeDegree: 
					NodeDegree[v] = 1
				else:
					NodeDegree[v]  +=1
			counter +=1
	print('Finish Processing, Start dumping')
	print('Total Nodes', len(NodeDegree))
	print('maxDegree', max(iter(NodeDegree.items()), key=operator.itemgetter(1))[1]) # operator.itemgetter(1): a calable object feteching the 1th element of an iteratble input.
	print('minDegree', min(iter(NodeDegree.items()), key=operator.itemgetter(1))[1]) 
	print('AverageDegree', sum(NodeDegree.values())/float(len(NodeDegree)))



	FinalNodeList =[]
	FinalNodeDegree  = {}
	max_degree = 6000
	min_degree = 0

	for key in NodeDegree:
		if min_degree <= NodeDegree[key] and NodeDegree[key] <= max_degree: # 在NodeDegree中选一部分出来
			FinalNodeList.append(key)
			FinalNodeDegree[key] = NodeDegree[key]


	print('\nTotal Nodes', len(FinalNodeList))
	print('maxDegree', max(iter(NodeDegree.items()), key=operator.itemgetter(1))[1])
	print('minDegree', min(iter(NodeDegree.items()), key=operator.itemgetter(1))[1]) 
	print('AverageDegree', sum(FinalNodeDegree.values())/float(len(FinalNodeDegree)))

	pickle.dump( FinalNodeList, open(save_dir+'NodesDegree'+str(max_degree)+'_'+str(min_degree)+'.list', "wb" ))

def worker(out_queue, sub_lines):
	print('pid:', os.getpid())
	for i in tqdm(range(len(sub_lines))):
		line = sub_lines[i]
		if line[0] != '#':
			u, v = list(map(int, line.split(' ')))
			out_queue.append([(u, v), 1])

def generate_subgraph_edge(save_dir, edge_file, scale):
	max_degree = 6000
	min_degree = 0

	NodeList = pickle.load(open(save_dir+'NodesDegree'+str(max_degree)+'_'+str(min_degree)+'.list', "rb" ))
	print('Done with loading List')

	NodeNum = len(NodeList)
	print("original nodes:", NodeNum)

	# sample a subset of total subnode list
	Small_NodeList = [NodeList[i] for i in sorted(random.sample(range(len(NodeList)), NodeNum//scale))] 
	NodeList = Small_NodeList
	print("sub nodes:", len(NodeList))
	pickle.dump(NodeList, open(save_dir+'Small_NodeList.list', "wb" ))

	start = time.time()
	G = nx.DiGraph()
	print('Start Reading')
	with open(edge_file) as f:
		lines = f.readlines()
	print('{} lines(edges)'.format(len(lines)))


	for i in tqdm(range(len(lines))):
		line = lines[i]
		if line[0] == '#':
			continue
	
		u, v = list(map(int, line.split()))
		if u in NodeList and v in NodeList: # NodeList is the small node list
			try:
				G[u][v]['weight'] += 1
			except:
				G.add_edge(u,v, weight=1)
			if 'Flickr' in save_dir or 'DBLP' in save_dir:
				try:
					G[v][u]['weight'] += 1 
				except:
					G.add_edge(v, u, weight=1)


	pickle.dump( G, open(save_dir+'Small_Final_SubG.G', "wb" ))
	print('\nDumped')
	print('number of nodes:', len(G.nodes()))
	print('number of edges:', len(G.edges()))
	print('Built sub graph G', time.time() - start, 's')

if __name__ == '__main__':
		
	# save_dir = '/home/xiawenwen/datasets/Flickr/'
	# flickr_file = os.path.join(save_dir, 'flickrEdges.txt')

	# save_dir = '/home/xiawenwen/datasets/NetHEPT/'
	# file = os.path.join(save_dir, 'Cit_HepTh.txt')

	# save_dir = '/home/xiawenwen/datasets/HEPPH/'
	# file = os.path.join(save_dir, 'Cit-HepPh.txt')

	save_dir = '/home/xiawenwen/datasets/DBLP/'
	file = os.path.join(save_dir, 'com-dblp.ungraph.txt')

	generate_subnode_list(save_dir, file)
	generate_subgraph_edge(save_dir, file, scale=10)
