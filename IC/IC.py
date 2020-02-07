''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'
from copy import deepcopy
from random import random
import numpy as np

def runICmodel_n (G, S, P):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    根据seeds和传播模型，进行传播，关键
    '''
    reward = 0
    T = deepcopy(S) # copy already selected nodes，保存最终会传播到的所有结点
    E = {}

    # ugly C++ version
    i = 0
    while i < len(T): # 这里T相当于一个队列了
        for v in G[T[i]]: # for neighbors of a selected node                
            w = G[T[i]][v]['weight'] # count the number of edges between two nodes，两个结点之间可能还有不止一条边？根据从数据集构建图的规则看是这样的。数据集Flickr中u向v发了一个image，就认为是增加一条边
            # if random() <= 1 - (1-P[T[i]][v]['weight'])**w: # if at least one of edges propagate influence
            if random() <= P[T[i]][v]['weight']:
                    # print T[i], 'influences', v
                if v not in T: # if it wasn't selected yet
                    T.append(v)
                if (T[i], v) in E:
                    E[(T[i], v)] += 1 # 这里实际上所有值都是1，不会出现大于1的情况，因为(T[i],v)这个键不会重复。E最多只存1，而不会超过1
                else:
                    E[(T[i], v)] = 1
        i += 1
    reward = len(T)

    return reward, T, E # 返回的E就是代表了edge-level的信息，edge的出发结点激活了edge的到达结点。如果使用node-level的反馈来更新，就没有edge的信息。

def runICmodel_node_feedback(G, S, P):
    ''' Runs independent cascade model.
    Input: 
        G: networkx graph object
        S: initial set of vertices
        p: propagation probability
    Output: 
        len(T): reward, number influenced vertices (including S)
        history: activation time of each activated node
        None: placeholder for "lived edges"
    '''

    T = deepcopy(S) # copy already selected nodes

    history = {} # 保存每一个节点的激活时间，只有被激活了的节点才会在这个字典里面
    for node in S:
        history[node] = 0


    activation_state = {} # 记录每一个节点的激活状态
    for node in list(G.nodes):
        if node not in S:
            activation_state[node] = False
        else:
            activation_state[node] = True


    activation_time = 0
    last_activated_set = S 
    while len(last_activated_set) > 0:
        current_activated_set = [] # 这句一定要清空current activation set
        activation_time += 1
        for u in last_activated_set:
            for v in G[u]: # neighbors of a selected node
                if activation_state[v] is False: # if it wasn't selected yet
                    w = G[u][v]['weight']
                    if random() <= 1 - (1-P[u][v]['weight'])**w: # if at least one of edges propagate influence
                    # if random() <= P[u][v]['weight']:
                        T.append(v)
                        current_activated_set.append(v)
                        history[v] = activation_time
                        activation_state[v] = True
        
        last_activated_set = deepcopy(current_activated_set)

    return len(T), history, None # 所有被激活了的结点，每一个被激活了的结点的激活时间

def runIC (G, S, p = .01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability，这个地方是将所有edge的probability的概率做相等看待，因此只有一个小p
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes
    E = {}

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
                    E[(T[i], v)] = 1
        i += 1

    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)

    return T#len(T), T, E

def runIC2(G, S, p=.01):
    ''' Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S) 
    Acur = deepcopy(S) # ？
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random.random() < 1 - (1-p)**w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        print(i, Anext)
        i += 1
        T.extend(Acur)
        Anext = []
    return T
    
def avgSize(G,S,p,iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G,S,p)))/iterations
    return avg
