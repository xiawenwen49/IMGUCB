import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import argparse
from scipy.stats import norm

def nonlin_f(val): 
	# Flickr
	if val <= 0.01:
		return val
	elif val <= 0.05:
		return 0.05 - val
	else:
		return val


def nonlin_hepph(val): 
	# Flickr
	val = val
	if val <= 0.01:
		return val
	elif val <= 0.2:
		return 0.2 - val
	else:
		return val

def nonlin_NetHEPT(influ, suscep, val): 
	l2 = np.linalg.norm(np.concatenate((influ, suscep)), ord=2)
	p = np.arctan(l2 * 1) * (2/np.pi)
	return p


class genMLE(object):
	def __init__(self, save_dir, G_file, dimension=4):
		'''
		load the small graph
		'''
		self.save_dir = save_dir
		self.G_file = G_file
		self.G = pickle.load(open(os.path.join(self.save_dir, self.G_file), 'rb'))
		self.Prob_filename = os.path.join(self.save_dir, 'Probability.dic')
		self.node_feature_filename = os.path.join(self.save_dir, 'Small_nodeFeatures.dic')
		self.dimension = dimension
		pass

	def generate_node_features(self):

		nodeDic = {}
		for u in self.G.nodes():
			# s = len(G.edges(u)) # out edges
			theta = np.random.uniform(-1, 1, size=(self.dimension,))
			beta = np.random.uniform(-1, 1, size=(self.dimension,))
			nodeDic[u] = [beta, theta] # [susceptibility, influence] 

		pickle.dump(nodeDic, open(self.node_feature_filename, "wb" )) # node vectors
		print("node fearures dump finished")
		pass

	def generate_edge_prob_from_vector(self):
		'''
		两向量做内积后取sigmoid的输出
		'''
		nodeDic = pickle.load(open(self.node_feature_filename, 'rb'))
		
		edgeDic = {}
		degree = []
		for u in self.G.nodes():
			d = 0
			for v in self.G[u]:
				prob = sigmoid( np.dot(nodeDic[u][1], nodeDic[v][0]) )
				edgeDic[(u,v)] = prob
				d += prob
			degree.append(d) # soft degree
		pickle.dump(edgeDic, open(self.Prob_filename, "wb" )) # edge probabilities
		print("edge probability dump finished")
		plt.hist(degree)
		plt.show()
		pass

	def generate(self):
		self.generate_node_features()
		self.generate_edge_prob_from_vector()
		pass

def sigmoid(x):
	x = x - 1
	return 1 / ( 1 + np.exp(-x) )  * 0.1

def featureUniform(dimension, scale):
	vector = np.array([random.random() for i in range(dimension)])
	l2_norm = np.linalg.norm(vector, ord =2)
	vector = vector/l2_norm

	gau = np.random.normal(0.5, 0.5, 1)[0] # gau?
	while gau < 0 or gau > 1:
		gau = np.random.normal(0.5, 0.5, 1)[0]
	vector = vector / scale * gau * 1.5
	
	return vector


def generate_node_features(dataset="NetHEPT"):
	assert dataset
	G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb'))
	dimension = 4
	nodeDic = {}
	for u in G.nodes():
		if dataset == "NetHEPT":
			s = 1
			if s == 0: s = 1
		else:
			s = len(G.edges(u)) # out edges
			if s == 0: s = 1
		nodeDic[u] = [featureUniform(dimension, 1), featureUniform(dimension, s)] # [susceptibility, influence]

	pickle.dump(nodeDic, open(save_dir+'Small_nodeFeatures.dic', "wb" )) # node vectors
	print('node feature Small_nodeFeatures.dic generated')


def generate_edge_prob_nonlinear(dataset):

	assert dataset == 'Flickr' or dataset == 'NetHEPT' or dataset == 'HEPPH' or dataset == 'DBLP' 
	G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb'))
	nodeDic = pickle.load(open(save_dir+'Small_nodeFeatures.dic', 'rb'))
	
	edgeDic = {}
	degree = []
	for u in G.nodes():
		d = 0
		for v in G[u]:
			influ = nodeDic[u][1]
			suscep = nodeDic[v][0]
			val = np.dot(influ, suscep)
			
			
			if dataset == 'Flickr':
				val = np.clip(val, 0, 1)
				prob = nonlin_f(val)
			elif dataset == 'NetHEPT':
				val = np.clip(val, 0, 1)
				prob = nonlin_NetHEPT(influ, suscep, val)
			elif dataset == 'DBLP':
				prob = nonlin_NetHEPT(influ, suscep, val)


			prob = np.clip(prob, 0, 1)

			edgeDic[(u,v)] = prob
			print(prob)
			d += prob
		degree.append(d) # soft degree
	pickle.dump(edgeDic, open(save_dir+'Probability_nonlinear.dic', "wb" )) # edge probabilities
	print("nonlinear edge probability Probability_nonlinear.dic generated")
	# plt.hist(degree)
	# plt.show()

def generate_edge_prob_from_vector():
	'''
	G: graph
	nodeDic: node features
	'''
	G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb'))
	nodeDic = pickle.load(open(save_dir+'Small_nodeFeatures.dic', 'rb'))
	
	edgeDic = {}
	degree = []
	for u in G.nodes():
		d = 0
		for v in G[u]:
			prob = np.dot(nodeDic[u][1], nodeDic[v][0]) * 0.2 	
			prob = np.clip(prob, 0, 1)		
			edgeDic[(u,v)] = prob
			print(prob)
			d += prob
		degree.append(d) # soft degree
	pickle.dump(edgeDic, open(save_dir+'Probability.dic', "wb" )) # edge probabilities


def generate_edge_prob_iid():
	'''generate edge probabilities from [0,1], independently. Hence the prob matrix is NOT a low rank matrix.
	'''
	edgeDic = {}
	G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb')) # load graph
	scale = 0.1
	for u in G.nodes():
		for v in G[u]:
			edgeDic[(u, v)] = np.random.random() * scale
	pickle.dump(edgeDic, open(save_dir + 'Edge_probability_iid.dic', "wb")) # save edge probability

def generate_edge_feature_vector():
	edgeDic = {}
	G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb')) # load graph
	nodeDic = pickle.load(open(save_dir+'Small_nodeFeatures.dic', 'rb')) # load node feature
	for u in G.nodes():
		for v in G[u]:
			edgeDic[(u,v)] = np.outer(nodeDic[u][1], nodeDic[v][0]).reshape(-1)
	pickle.dump(edgeDic, open(save_dir+'Small_edgeFeatures.dic', "wb" )) # save edge vector
	print("edge features generated.")



def plot_non_liear_func(func=nonlin_f):
	x = np.linspace(0, 0.1, num=1000)
	y = [func(val) for val in x]
	
	f, ax = plt.subplots()
	ax.grid(True)
	ax.scatter(x, y, color='black', s=2 )
	# ax.plot(x, y, color='black')
	f.savefig('non_linear_plot.pdf')

save_dir = '/home/xiawenwen/datasets/DBLP/'


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-g', type=str)
args = parser.parse_args()
if __name__ == "__main__":
	if args.g == 'node':
		generate_node_features()

	elif args.g == 'edge':
		generate_edge_feature_vector()

	elif args.g == 'prob':
		generate_edge_prob_from_vector()
	
	elif args.g == 'nlprob': # non-linear prob
		generate_edge_prob_nonlinear(dataset='DBLP')

	elif args.g == 'plotnon':
		plot_non_liear_func()

