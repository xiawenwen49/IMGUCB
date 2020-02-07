import time
import os
import pickle 
import pandas
import datetime
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import sys

from conf import *
from Tool.utilFunc import *
import utilis.plot
import logging
logging.basicConfig(level=logging.INFO)

from BanditAlg.CUCB import UCB1Algorithm
from BanditAlg.greedy import eGreedyAlgorithm 
from BanditAlg.IMFB import MFAlgorithm
from BanditAlg.IMLinUCB import N_LinUCBAlgorithm 
from BanditAlg.IMFB_MLE import IM_mle
from BanditAlg import IMGaussianUCB

from IC.IC import runICmodel_n, runICmodel_node_feedback
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp
from Oracle.random import random_seeds

class simulateOnlineData:
    def __init__(self, G, P, oracle, seed_size, iterations, dataset, algorithms, record=True, show=True, resdir=None):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.actual_iters = 0 
        self.dataset = dataset
        self.algorithms = algorithms
        self.record = record
        self.show = show
        self.resdir = resdir if resdir != None else "./SimulationResults"

        
        self.startTime = datetime.datetime.now()
        self.tttmmm = time.time()
        self.startTimeStr = self.startTime.strftime('%m_%d_%H_%M_%S') 
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.Loss = {}
        self.oracle_reward = []

    def runAlgorithms(self):

        for alg_name, alg in list(self.algorithms.items()):
            self.AlgReward[alg_name] = [] 
            self.BatchCumlateReward[alg_name+'_cum'] = []
            self.Loss[alg_name] = []
        
        for iter_ in range(self.iterations):
            optS = self.oracle(self.G, self.seed_size, self.TrueP)
            
            rand_S = random_seeds(self.G, self.seed_size, self.TrueP)
            
            logging.info("\n")
            logging.info("Iter {}/{}".format(iter_+1, self.iterations))

            optimal_reward, live_nodes, live_edges = runICmodel_n(G, optS, self.TrueP)
            rand_S_reward, _, _ = runICmodel_n(G, rand_S, self.TrueP)

            self.oracle_reward.append(optimal_reward)
            
            for alg_name, alg in list(self.algorithms.items()): 
                S = alg.decide()

                if alg_name == 'IMFB_MLE':
                    epsilon = 1
                    ga = 1
                    if random.random() < epsilon: 
                        rand_S = random_seeds(alg.G, alg.seed_size, alg.currentP)
                        S = rand_S
                    epsilon = epsilon*ga
                    reward, live_nodes, live_edges = runICmodel_node_feedback(G, S, self.TrueP)
                    alg.updateParameters(S, live_nodes, live_edges, iter_)
                    S = alg.decide() # 再重新用oracle选seed
                    reward, live_nodes, live_edges = runICmodel_node_feedback(G, S, self.TrueP)
                elif alg_name == 'IMFB':
                    reward, live_nodes, live_edges = runICmodel_n(G, S, self.TrueP)
                    alg.updateParameters(S, live_nodes, live_edges, iter_)
                elif alg_name == 'IMGUCB':
                    reward, live_nodes, live_edges = runICmodel_n(G, S, self.TrueP) 
                    alg.updateParameters(S, live_nodes, live_edges, iter_)
                else:
                    reward, live_nodes, live_edges = runICmodel_n(G, S, self.TrueP) 
                    alg.updateParameters(S, live_nodes, live_edges, iter_)


                self.AlgReward[alg_name].append(reward)
                self.BatchCumlateReward[alg_name+'_cum'].append(self.BatchCumlateReward[alg_name+'_cum'][-1]+reward if iter_ > 0 else reward)
                self.Loss[alg_name] = alg.getLoss()

                logging.info("{}: reward:{}, loss:{}".format(alg_name, reward, self.Loss[alg_name][-1]))
                
            logging.info("{}:{}".format('oracle', optimal_reward))
            logging.info("{}:{}".format('random seed baseline', rand_S_reward))
            logging.info('total time: %.2f' % (time.time() - self.tttmmm))
            
            self.actual_iters += 1 
            if self.record:
                self.resultRecord(iter_) 
            if self.show:
                self.showResult(iter_) 
        

    def resultRecord(self, iter_=None, mod=5):
        '''
        保存reward数据： self.AlgReward、self.BatchCumlateReward这两个字典
        '''
        if not ((iter_+1) % mod == 0 or (iter_+1) == self.iterations ):
            return

        info = '{}_seedsize{}_{}_{}_{}'.format(self.startTimeStr, self.seed_size, str(self.oracle.__name__), self.dataset, '_'.join(self.algorithms.keys()))
        filepath = os.path.join(self.resdir, 'reward')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = 'Reward_{}{}'.format(info, '.csv')
        save_dic = {**self.AlgReward, **self.BatchCumlateReward} 
        df = pandas.DataFrame.from_dict(save_dic, orient="columns")
        df.to_csv(os.path.join(filepath, filename), index_label='Time(Iteration)')

        # 保存loss
        filepath = os.path.join(self.resdir, 'loss')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = 'Loss_{}{}'.format(info, '.csv')
        df = pandas.DataFrame.from_dict(self.Loss, orient="columns")
        df.to_csv(os.path.join(filepath, filename), index_label='Time(Iteration)')

    def showResult(self, iter_, mod=5):
        if not ((iter_+1) % mod == 0 or (iter_+1) == self.iterations ): 
            return
        info = '{}_seedsize{}_{}_{}_{}'.format(self.startTimeStr, self.seed_size, str(self.oracle.__name__), self.dataset, '_'.join(self.algorithms.keys()))
        
        plot_obj = utilis.plot.Plot()
        plotArgs = [{'color':'red'}, {'color':'blue'}, {'color':'green'}, {'color':'cyan'}, {'color':'magenta'}, {'color':'black'}]
        num_algs = len(list(self.algorithms.keys()))
        plotArg_list = plotArgs[:num_algs]

        x_list = [list(range(1, self.actual_iters+1)) for i in range(num_algs)]
        legend_list = list(self.AlgReward.keys())
        y_list = [self.AlgReward[key] for key in legend_list]
        xlabel = 'Iteration'
        ylabel = 'Reward'
        title = 'Round reward'
        
        filepath = os.path.join(self.resdir, 'reward')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = 'Reward_{}{}'.format(info, '.pdf')
        
        plot_obj.plot_data_lists(y_list, x_list, legend_list, title, os.path.join(filepath, filename), xlabel=xlabel,
                                ylabel=ylabel,
                                length=10,
                                height=7,
                                label_fsize=10,
                                plotArg_list=plotArg_list)
        
        x_list = [list(range(1, self.actual_iters+1)) for i in range(num_algs)]
        y_list = [self.algorithms[key].getLoss() for key in legend_list]
        xlabel = 'Iteration'
        ylabel = 'Loss'
        title = 'Round loss'

        filepath = os.path.join(self.resdir, 'loss')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = 'Loss_{}{}'.format(info, '.pdf')


        plot_obj.plot_data_lists(y_list, x_list, legend_list, title, os.path.join(filepath, filename), xlabel=xlabel,
                                ylabel=ylabel,
                                length=10,
                                height=7,
                                label_fsize=10,
                                plotArg_list=plotArg_list)
        print('iter {} figure saved'.format(iter_))



parser = argparse.ArgumentParser(description=None)
parser.add_argument('-imgucb', action='store_true')
parser.add_argument('-imfbmle', action='store_true')
parser.add_argument('-imfb', action='store_true')
parser.add_argument('-imlinucb', action='store_true')
parser.add_argument('-egreedy', action='store_true')
parser.add_argument('-ucb1', action='store_true')
parser.add_argument('-all', '--all_algs', action='store_true')
parser.add_argument('-repeat', type=int, default=1, required=True)
parser.add_argument('-resdir', type=str, required=True)
parser.add_argument('-dataset', type=str, choices=['Flickr', 'NetHEPT', 'HEPPH', 'DBLP'], required=True)
parser.add_argument('-prob', type=str, choices=['linear', 'nonlinear'], required=True) # linear or nonlinear prob

args = parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    if args.dataset == 'Flickr':
        data_dir_ = Flickr_data_dir
    elif args.dataset == 'NetHEPT':
        data_dir_ = NetHEPT_data_dir
    elif args.dataset == 'HEPPH':
        data_dir_ = HEPPH_data_dir
    elif args.dataset == 'DBLP':
        data_dir_ = DBLP_dir

    def join_(file):
        return os.path.join(data_dir_, file)

    G = pickle.load(open(join_(graph_address), 'rb'), encoding='latin1')
    if args.prob == 'nonlinear':
        prob = pickle.load(open(join_(nonlinear_prob_address), 'rb'), encoding='latin1')
    elif args.prob == 'linear':
        prob = pickle.load(open(join_(linear_prob_address), 'rb'), encoding='latin1')
        
    node_vector = pickle.load(open(join_(node_feature_address), 'rb'), encoding='latin1')
    

    trueP = nx.DiGraph()
    for (u,v) in G.edges():
        trueP.add_edge(u, v, weight=prob[(u,v)]) 
    print('Done with Loading Feature')
    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Graph build time: %.2f' % (time.time() - start) )

    for i in range(args.repeat):
        print("Repeat {}".format(i+1))
        algorithms = {}
        if args.imgucb:
            algorithms['IMGUCB'] = IMGaussianUCB.IMGaussianUCB(G, trueP, node_vector, seed_size, oracle, node_dim=node_dimension, edge_dim=edge_dimension,kernel_size=kernel_size, sigma=sigma, gamma=gamma, var=var, c=c)
        if args.imfb:
            algorithms['IMFB'] = MFAlgorithm(G, trueP, node_vector, seed_size, oracle, node_dimension)
        if args.imfbmle:
            algorithms['IMFB_MLE'] = IM_mle(G, trueP, node_vector, seed_size, oracle, node_dimension)
        if args.imlinucb:
            edge_vector = pickle.load(open(join_(edge_feature_address), 'rb'), encoding='latin1')
            algorithms['LinUCB'] = N_LinUCBAlgorithm(G, trueP, node_vector, seed_size, oracle, node_dimension*node_dimension, alpha_1, lambda_, edge_vector, 1) # 这个需要edge feature? feature_dic
        if args.egreedy:
            algorithms['egreedy_0.1'] = eGreedyAlgorithm(G, trueP, seed_size, oracle, 0.1)
        if args.ucb1:
            algorithms['UCB1'] = UCB1Algorithm(G, trueP, node_vector, seed_size, oracle)

        simExperiment = simulateOnlineData(G, trueP, oracle, seed_size, iterations, args.dataset, algorithms, record=True, show=True, resdir=args.resdir)
        simExperiment.runAlgorithms()
