import torch
import numpy as np
import networkx as nx 
import random
import pickle
from copy import deepcopy
from tqdm import tqdm

class MLENodeStruct(object):
    def __init__(self, dimension, userID):
        self.userID = userID
        self.dim = dimension

        self.theta = np.random.random(size=self.dim)
        self.beta = np.random.random(size=self.dim)

        pass

def compute_L(Ac_tensor, Bc_tensor, s_v, dimension=4):
    
    s_v = s_v.reshape((-1,1))
    if Ac_tensor.shape[0] == 0:
        term1 = 0
    else:
        term1 = (torch.log(torch.ones(dimension) - torch.sigmoid(torch.mm(Ac_tensor, s_v)) )).sum()
    term2 = torch.log(1 - torch.exp( (torch.log( torch.ones(len(Bc_tensor)) - torch.sigmoid(torch.mm(Bc_tensor, s_v)) ) ).sum() ) ) # 相乘：先求log，再相加，再求exp
    log_Likelihood = -1 *(term1 + term2)
    # log_Likelihood = term1 + term2
    
    return log_Likelihood
    

def optimize(Ac_array, Bc_array, s_v, dimension=4):
    '''
    input: numpy array
        Ac_array: theta array of nodes in Ac set 
        Bc_array: theta array of nodes in Bc set
        s_v: beta (susceptibility) array of node v
    
    return: numpy array
        updated arrays
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iteration = 1
    Ac_tensor = torch.tensor(Ac_array, dtype=torch.float, requires_grad=True)#.cuda(device) # construct tensor
    Bc_tensor = torch.tensor(Bc_array, dtype=torch.float, requires_grad=True)#.cuda(device)
    s_v_tensor = torch.tensor(s_v, dtype=torch.float, requires_grad=True)#.cuda(device)

    if Ac_tensor.shape[0] > 0: 
        optimizer = torch.optim.SGD([Ac_tensor, Bc_tensor, s_v_tensor], lr=0.01, momentum=0.9)
        
        for i in range(iteration): 
            optimizer.zero_grad()
            log_L = compute_L(Ac_tensor, Bc_tensor, s_v_tensor, dimension) # compute log lokelihood
            # print("opt step:{}".format(i), log_L)
            log_L.backward(retain_graph=True)
            optimizer.step()

        return Ac_tensor.detach().numpy(), Bc_tensor.detach().numpy(), s_v_tensor.detach().numpy().reshape(1,-1)

    else: 
        optimizer = torch.optim.SGD([Ac_tensor, Bc_tensor, s_v_tensor], lr=0.01, momentum=0.9)


        for i in range(iteration):
            optimizer.zero_grad()
            log_L = compute_L(Ac_tensor, Bc_tensor, s_v_tensor)
            # print("opt step:{}".format(i), log_L)
            log_L.backward(retain_graph=True)
            optimizer.step()

        return np.zeros((0, dimension)), Bc_tensor.detach().numpy(), s_v_tensor.detach().numpy().reshape(1,-1)



class IM_mle(object):
    def __init__(self, G, P, parameter, seed_size, oracle, dimension, feedback='node', **kwargs):
        self.G = G
        self.trueP = P
        self.parameter = parameter # true node features
        self.oracle = oracle
        self.seed_size = seed_size
        self.q = 0.25

        self.dimension = dimension
        self.feedback = feedback
        self.list_loss = []

        self.currentP =nx.DiGraph()
        self.users = {}  # Nodes

        for u in self.G.nodes(): 
            self.users[u] = MLENodeStruct(dimension, u)
    
        for u in self.G.nodes(): 
            for v in self.G[u]:
                self.currentP.add_edge(u,v, weight=getP(self.users[u].theta, self.users[v].beta) )
    
    def construct_vector_list(self, nodes, theta=True):
        '''从self.user中提取特定的node的向量，组成二维array
        '''
        array = np.zeros(shape=(len(nodes), self.dimension))
        if theta is True: 
            for i, n in enumerate(nodes):
                array[i] = self.users[n].theta
        else:
            for i, n in enumerate(nodes):
                array[i] = self.users[n].beta
        return array


    def writeback_vector_array(self, nodes, updated_tensor, theta=True):
        '''
        nodes: list
        updated_tensor: np array!
        '''
        assert len(nodes) == updated_tensor.shape[0], "number of nodes not equal"
        if theta is True:
            for i, node in enumerate(nodes):
                self.users[node].theta = deepcopy(updated_tensor[i]) # deepcopy!!
        else:
            for i, node in enumerate(nodes):
                self.users[node].beta = deepcopy(updated_tensor[i])

    def decide(self):
        '''choose a seed set'''
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S

    def updateParameters(self, S, history, live_edges=None, iter_=None):
        '''更新node embedding，依据node-level的feedback
        计划用torch计算likelihood对node embedding的gradient
        1. 确定哪些node的vector需要更新，构建torch tensor
        2. 根据这些node的激活了的父节点的激活时刻，构建log likelihood函数
        3. 计算LL函数对这些node的vector的梯度
        4. 更新torch tensor，将更新后的tensor向量写回到self.user中
        '''
        loss_p = 0
        loss_theta = 0 # loss_out
        loss_beta = 0 # loss_in
        count = 0
        for node in tqdm(list(history.keys())):
            if history[node] > 0:
                Ac_set, Bc_set = self.find_activated_parents(node, history)
                Ac_array = self.construct_vector_list(Ac_set)
                Bc_array = self.construct_vector_list(Bc_set)
                s_v = self.users[node].beta
                
                Ac_updated, Bc_updated, s_v_updated = optimize(Ac_array, Bc_array, s_v, dimension=self.dimension)
                
                
                self.writeback_vector_array(Ac_set, Ac_updated)
                self.writeback_vector_array(Bc_set, Bc_updated)
                self.writeback_vector_array([node], s_v_updated, theta=False)

                
                for pre in Ac_set + Bc_set:
                    p = getP(self.users[pre].theta, self.users[node].beta)
                    self.currentP[pre][node]['weight'] = p

                    loss_p += np.abs(p-self.trueP[pre][node]['weight'])
                    
                    loss_theta += np.linalg.norm(self.users[pre].theta - self.parameter[pre][1], ord=2) # [susceptibility(beta), influence(theta)]
                    loss_beta += np.linalg.norm(self.users[node].beta - self.parameter[node][0], ord=2)

                    count += 1
        
        self.list_loss.append([loss_p/count, loss_theta/count, loss_beta/count])
        # print("loss:", self.list_loss[-1])
                

    def find_activated_parents(self, node, history):
        '''
        return:
            Ac_set: activated parents at time t-2, t-3, ... 0
            Bc_set: activated parents at time t-1
        '''
        Ac_set = []
        Bc_set = []
        parents = self.G.predecessors(node)
        t = history[node]
        for p in parents:
            if history.get(p, False) is not False:
                if history[p] == t-1:
                    Bc_set.append(p)
                if history[p] <= t-2:
                    Ac_set.append(p)
                    pass
        
        return Ac_set, Bc_set

    def getLoss(self):
        return np.array(self.list_loss)
    
def getP(theta, beta):
    assert len(theta) == len(beta), "length of theta and beta not equal" 
    return 1 / ( 1 + np.exp(-1*np.dot(theta, beta) ) )
