import numpy as np 

def random_seeds(G, k, Ep):
    nodes = list(G.nodes)
    indexs = np.random.randint(0, len(nodes), size=k)
    S = [nodes[i] for i in indexs]
    
    return S
