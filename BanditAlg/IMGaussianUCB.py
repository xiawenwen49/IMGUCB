import numpy as np 
import numexpr as ne
import heapq
import networkx as nx
import tqdm
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import logging
logging.basicConfig(level=logging.INFO)

class UserStruct(object):
    def __init__(self, userID, vector):
        self.userID = userID
        self.x = vector
        pass

class IMGaussianUCB(object):
    ''' Gaussian process UCB algorithm for semi-bandit IM '''

    def __init__(self, G, true_prob, node_vector, seed_size, oracle, node_dim, edge_dim, 
                kernel_size=1000, 
                sigma=1, 
                gamma=0.1, 
                var=1, 
                c=0.2):
        """
        kernel_size: number of samples to compute the Σ matrix, 用来做GP的样本数
        sigma: kernel matrix里面乘到单位阵上的系数
        gamma, var: RBF函数的参数。K(x1, x2) = var * exp(-gamma * ||x1 - x2||^2)
        c: 计算upper bound时的variance的倍数，μ + c*var
        """
        self.G = G
        self.trueP = true_prob # networkx graph类型
        self.node_vector = node_vector # dict
        # self.edge_vector = edge_vector # dict
        self.seed_size = seed_size
        self.oracle = oracle
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.sigma = sigma 
        self.gamma = gamma
        self.var = var
        self.c = c
        self.kernel_size = kernel_size # 暂定n=5000
        self.kernelF = self.rbfKernel

        self.decayFactor = 1
        self.decay = 0.95

        self.loss = [] # 每次迭代之后的loss。没有loss_in, loss_out，只有loss_p，在每条边上。
        self.edgeCounter = {} # 每条边被记录到的次数
        self.edgeP = {} # 根据每条边的历史观测记录的历史均值概率
        
        self.trueP_mean = self.get_ntx_mean(self.trueP)
        for u in self.G.nodes(): # node的原始编号
            for v in self.G[u]:
                self.edgeCounter[(u,v)] = 0
                self.edgeP[(u,v)] = 0

        self.n_edges = len(self.G.edges())
        self.topKedges = [] # top K edges. 被观测到频次最多的topK条边，构成tuple list: [(u1,v1), (u2,v2), ...]。用来构成Σ矩阵。这里也许有替代方案。
        self.KernelMat = None # Kernel matrix, 这个矩阵是需要保存的

        # current P estimation
        self.ucbP = nx.DiGraph() # UCB估计（均值估计+upper bound）
        self.miuP = {} # 均值/期望(不加upper bound的）估计
        for u in self.G.nodes():
            for v in self.G[u]:
                self.ucbP.add_edge(u, v, weight=np.random.random())
                # self.ucbP.add_edge(u, v, weight=1.0)
                self.miuP[(u,v)] = np.random.random()
    
    def decide(self):
        S = self.oracle(self.G, self.seed_size, self.ucbP) # 暂且这么写
        return S

    def extract_edge_vectors(self, edge_keys):
        X = np.zeros((len(edge_keys), self.edge_dim) )
        Y = np.zeros((len(edge_keys), 1))
        for i, (u,v) in enumerate(edge_keys):
            X[i] = self.genEdgeVector(self.node_vector[u], self.node_vector[v]).reshape(-1) # (u,v)是边的键，值是边的vector
            Y[i] =  self.edgeP[(u,v)]
        return X, Y
    
    def updateParameters_sklearn(self, live_nodes, live_edges, iter_):
        # 更新GP模型
        # kernel = 1.0**2 * RBF(length_scale=1.0)
        kernel = RBF()
        self.GP = GaussianProcessRegressor(kernel=kernel, alpha=self.sigma, normalize_y=True) # 默认带optimizer
        self.GP.fit(self.X, self.Y) # 拟合

        # 计算边的估计与upper bound
        edge_matrix = []
        # for u in live_nodes:
        for u in self.G.nodes():
            for v in self.G[u]:
                x = self.genEdgeVector(self.node_vector[u], self.node_vector[v])
                edge_matrix.append(x)
        edge_matrix = np.array(edge_matrix)
        print('{} edges p need to be computed'.format(edge_matrix.shape[0]))

        # print('edge matrix shape:', edge_matrix.shape)
        num_process = 10
        p = Pool(num_process)
        indices = np.arange(1, num_process) * (edge_matrix.shape[0]//num_process) # 起始位置
        edge_matrix_split = np.split(edge_matrix, indices, axis=0)

        result_list = p.map(self.predict_miu_sig_skl, edge_matrix_split)
        p.close()  
        p.join() 

        estimate_mat = np.vstack(result_list)

        # 更新边的估计与upper bound
        index = 0
        # for u in live_nodes:
        for u in self.G.nodes():
            for v in self.G[u]:
                miu, std = estimate_mat[index][0], estimate_mat[index][1]
                miu = np.clip(miu, 0.0, 1.0)
                upp = np.clip(miu + self.decayFactor * self.c*std, 0.0, 1.0 )
                self.miuP[(u,v)], self.ucbP[u][v]['weight'] = miu, upp
                index += 1

    def predict_miu_sig_skl(self, X):
        """
        用sklearn的GP计算miu与sigma
        """
        miu_var_mat = np.zeros((X.shape[0], 2))

        
        # for i, x in enumerate(X):
        #     x = x.reshape(1,-1)
        #     miu, std = self.GP.predict(x, return_std=True)
        #     miu_var_mat[i][0], miu_var_mat[i][1] = miu, std

        y_mean, y_std = self.GP.predict(X, return_std=True, return_cov=False)
        miu_var_mat[:, 0] = y_mean.flatten()
        miu_var_mat[:, 1] = y_std.flatten()

        return miu_var_mat
            

    def updateParameters(self, S, live_nodes, live_edges, iter_, mod=10, us_sklearn=True):

        for u in live_nodes:
            for v in self.G[u]:
                if (u,v) in live_edges:
                    reward = 1 
                else: reward = 0
                self.edgeCounter[(u,v)] += 1
                self.edgeP[(u,v)] = (self.edgeP[(u,v)]*(self.edgeCounter[(u,v)]-1) + reward) / self.edgeCounter[(u,v)]

        if (iter_ + 1) % mod == 0 and iter_ != 0: 
            self.topKedges = self.TopKkeys(self.kernel_size, self.edgeCounter)
            self.X, self.Y = self.extract_edge_vectors(self.topKedges)
            
            if us_sklearn:
                print('learning')
                self.updateParameters_sklearn(live_nodes, live_edges, iter_)
            else:
                print('using self implementation')
                self.KernelMat = self.computeKerMat()
                self.updateEdgeEstimate(live_nodes, live_edges)
            
            self.decayFactor *= self.decay
        
        self.computeLoss(live_nodes, live_edges)

        logging.info('mean miu: %f' % np.mean(list(self.miuP.values())))
        logging.info('mean ucb: %f' % self.get_ntx_mean(self.ucbP))
        logging.info('mean hisorically recorded p: %f' % np.mean(list(self.edgeP.values())))
        logging.info('true p mean: %f' % self.trueP_mean)
        logging.info('c=%f' % self.c)
        
    def get_ntx_mean(self, G_):
        sum_ = 0
        count = 0
        for u in self.G.nodes():
            for v in self.G[u]:
                sum_ += G_[u][v]['weight']
                count += 1

        return sum_ / count

    def TopKkeys(self, n, dic):
        topKkeys = heapq.nlargest(n, dic, key=dic.get)
        return topKkeys

    def TopKkeys_proportional(self, N):
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        hist, bin_edges = np.histogram(list(self.edgeP.values()), bins=bins )
        nums = np.ceil( hist/self.n_edges * N )
        nums = np.clip(nums, 100, 2000).astype(np.int32)

        partion_dics = [{} for i in range(10)] 

        for key, val in self.edgeP.items():
            index = int( np.clip(val, 0, 0.99)//0.1 )
            partion_dics[index][key] = self.edgeCounter[key]
        
        result_keys = []
        for i, ni in enumerate(nums):
            list_ = list(partion_dics[i].keys())
            if len(list_) <= ni:
                result_keys.extend(list_)
            else:
                list_ = heapq.nlargest(ni, partion_dics[i], key=partion_dics[i].get)
                result_keys.extend(list_)
        logging.info("matrix size:{}".format(len(result_keys)))
        return result_keys

    def TopKkeysDis(self, n, dic):
        pass

    def genEdgeVector(self, x1, x2):

        return np.concatenate((x1[1], x2[0])).reshape(-1)


    def updateEdgeEstimate(self, live_nodes, live_edges):
        edge_matrix = []
        edge_keys = []
        # for u in tqdm.tqdm(live_nodes):
        for u in live_nodes:
            for v in self.G[u]:
                x = self.genEdgeVector(self.node_vector[u], self.node_vector[v])
                edge_matrix.append(x)
                edge_keys.append((u,v))
        print('{} edges p need to be computed'.format(len(edge_keys)))
        edge_matrix = np.array(edge_matrix)
        
        miu_mat = self.predict_miu(edge_matrix)
        
        num_process = 20
        p = Pool(num_process)

        indices = np.arange(num_process) * (edge_matrix.shape[0]//num_process) 
        edge_matrix_split = np.split(edge_matrix, indices, axis=0)

        result_list = p.map(self.predict_sigma, edge_matrix_split)
        p.close()  
        p.join() 

        sig_mat = np.vstack(result_list)
        
        estimate_mat = miu_mat + self.c * np.diag(sig_mat).reshape(-1,1)
        estimate_mat = np.clip(estimate_mat, 0.0, 1.0)
        miu_mat = np.clip(miu_mat, 0.0, 1.0)
        
        index = 0
        for u in live_nodes:
        # for u in self.G.nodes():
            for v in self.G[u]:
                self.ucbP[u][v]['weight'] = estimate_mat[index][0]
                self.miuP[(u,v)] = miu_mat[index][0]
                index += 1

    def computeLoss(self, live_nodes, live_edges):
        
        count = 0
        loss = 0
        # for u in live_nodes:
        for u in self.G.nodes():
            for v in self.G[u]:
                loss += np.abs( self.miuP[(u,v)] - self.trueP[u][v]['weight'] )  
                count += 1

        self.loss.append(loss/count)

    def computeKerMat(self):

        KernelMat = np.linalg.inv( self.kernelF(self.X, self.X, self.gamma, self.var) + self.sigma * np.eye(self.kernel_size) )
        return KernelMat


    def predict_miu(self, x):
        # x = np.array(x).reshape((1, -1))
        miu = np.linalg.multi_dot( [self.kernelF(x, self.X, self.gamma, self.var), self.KernelMat, self.Y] )
        # miu = miu.flatten()[0]
        return miu

    def predict_sigma(self, X):
        sig_mat = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            x = np.array(x).reshape((1, -1))
            sig_sq = self.kernelF(x, x, self.gamma, self.var) + self.sigma * np.eye(x.shape[0]) \
                - np.linalg.multi_dot( [self.kernelF(x, self.X, self.gamma, self.var), self.KernelMat, self.kernelF(self.X, x, self.gamma, self.var)] )
 
            sig = np.sqrt(sig_sq.flatten()[0])

            sig_mat[i] = sig
        
        return sig_mat

    def rbfKernel(self, X1, X2, gamma, var):
        '''
        X1, X2是矩阵，每一行代表一个sample，所以返回的结果是X1.shape[0]*X2.shape[0]的一个矩阵
        实现，依据如下等式：
        K[i,j] = var * exp(-gamma * ||X1[i] - X2[j]||^2)
        i.e. ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        '''
        X1_norm = np.sum(X1**2, axis=-1)
        X2_norm = np.sum(X2**2, axis=-1)
        K = ne.evaluate('v * exp(-g * (A + B - 2 * C) )', {
            'A': X1_norm[:, None],
            'B': X2_norm[None, :],
            'C': np.dot(X1, X2.T),
            'g': gamma,
            'v': var
        })

        return K

    def getLoss(self):
        return np.array(self.loss)
     

if __name__ == "__main__":
    pass