
import numpy as np
import os
import pickle
import graphvite as gv 
import graphvite.application as gap 
import argparse


def generate_node_embedding(graph_address, model_name, embed_dim, node_embedding_path, node_embedding_inverse_path, inverse=False ):
    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    edges = list(G.edges)
    if inverse:
        edges_str = list(map(lambda x:(str(x[1]), str(x[0])), edges))
        node_embedding_path = node_embedding_inverse_path
    else:
        edges_str = list(map(lambda x:(str(x[0]), str(x[1])), edges)) # node必须用str来标识

    graph = gv.graph.Graph()
    graph.load(edges_str, as_undirected=False)

    solver = gv.solver.GraphSolver(dim=embed_dim) # https://graphvite.io/docs/latest/api/solver.html
    solver.build(graph)
    models = ['DeepWalk', 'LINE']
    solver.train(model=model_name, num_epoch=20000, shuffle_base=10)
    # save node embeddings
    print('saving...')
    node_embeddings = {}
    for u in G.nodes():
        index = graph.name2id[str(u)]
        embedding = solver.vertex_embeddings[index]

        l2_norm = np.linalg.norm(embedding, ord =2)
        scale = G.out_degree(u) + 1
        embedding = embedding/l2_norm * 1.5
        node_embeddings[u] = embedding

    
    pickle.dump(node_embeddings, open(node_embedding_path, "wb"))
    print('node embedding saved!')
    print('num of nodes:', len(G.nodes()))
    print('node embedding shape:', solver.vertex_embeddings.shape)


def generate_edge_embedding():
    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    node_embeddings = pickle.load(open(node_embedding_path, "rb"))
    edge_embeddings = {}

    for (u,v) in G.edges:
        edge_embeddings[(u,v)] = np.concatenate((node_embeddings[u], node_embeddings[v]), axis=0 )
    
    pickle.dump(edge_embeddings, open(edge_embedding_path, "wb"))
    print('edge embedding saved!')


def non_linear(p):
    if 0.01<= p < 0.25:
        p = 0.25 - p
    elif p > 0.5:
        p = 0
    return p

def func_p(alpha, beta, nonlinear=False, scale=2):
    if nonlinear:
        p = np.dot(alpha, beta) * 1
        p = non_linear(p)
    else:
        p = np.dot(alpha, beta) * 1
    p = np.clip(p, 0, 1)
    return p



def generate_probability(save_dir, graph_address, node_embedding_path, node_embedding_inverse_path, nonlinear=False):
    G = pickle.load(open(graph_address, 'rb'))
    node_embedding = pickle.load(open(node_embedding_path, 'rb'))
    node_embedding_inv = pickle.load(open(node_embedding_inverse_path, 'rb'))

    edgeDic = {}
    
    degree = []
    for u in G.nodes():
        d = 0
        for v in G[u]:
            alpha = node_embedding[u]
            beta = node_embedding_inv[v]
            prob = func_p(alpha, beta, nonlinear=nonlinear)
            edgeDic[(u,v)] = prob
            print(prob)
            d += prob
        degree.append(d) # soft degree
    file = save_dir+'Embedding_Probability_nonlinear.dic' if non_linear else save_dir+'Embedding_Probability_linear.dic'
    pickle.dump(edgeDic, open(file, "wb" ))
    print('mean prob:', np.mean(list(edgeDic.values())))


def dump_node_feature(save_dir, graph_address, node_embedding_path, node_embedding_inverse_path):
    G = pickle.load(open(graph_address, 'rb'))
    node_embedding = pickle.load(open(node_embedding_path, 'rb'))
    node_embedding_inv = pickle.load(open(node_embedding_inverse_path, 'rb'))

    nodeDic = {}
    for u in G.nodes():
        nodeDic[u] = [node_embedding_inv[u], node_embedding[u]]
    pickle.dump(nodeDic, open(save_dir+'Small_nodeFeatures_embed.dic', "wb" )) # node vectors

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-action', type=str)
parser.add_argument('-nonlinear', action='store_true')
args = parser.parse_args()

if  __name__ == "__main__":
        
    save_dir = '/home/xiawenwen/datasets/NetHEPT/'
    model_name = 'LINE'
    graph_address = os.path.join(save_dir, 'Small_Final_SubG.G')
    node_embedding_path = os.path.join(save_dir, model_name + '_node_embedding.dic')
    edge_embedding_path = os.path.join(save_dir, model_name + '_edge_embedding.dic')
    node_embedding_inverse_path = os.path.join(save_dir, model_name + '_node_embedding_inverse.dic')
    embed_dim = 32 # 32, 64, 128, 256, 512
    
    if args.action == 'embedding':
        generate_node_embedding(graph_address, model_name, embed_dim, node_embedding_path, node_embedding_inverse_path, inverse=False)
        generate_node_embedding(graph_address, model_name, embed_dim, node_embedding_path, node_embedding_inverse_path, inverse=True)    
        dump_node_feature(save_dir, graph_address, node_embedding_path, node_embedding_inverse_path)
    if args.action == 'prob':
        generate_probability(save_dir, graph_address, node_embedding_path, node_embedding_inverse_path, nonlinear=args.nonlinear)