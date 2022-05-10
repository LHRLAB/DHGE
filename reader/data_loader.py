
from __future__ import print_function
from __future__ import division

import logging
import torch
import numpy as np
import time
import numpy as np
import scipy.sparse as sp

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

def get_edge_labels(max_n):
    edge_labels = []
    max_seq_length=2*max_n-1
    max_aux = max_n - 2

    edge_labels.append([0, 1, 0] + [0,0] * max_aux)
    edge_labels.append([1, 0, 2] + [3,0] * max_aux)
    edge_labels.append([0, 2, 0] + [0,0] * max_aux)
    for idx in range(max_aux):
        edge_labels.append(
            [0, 3, 0] + [0,0] * idx + [0,4] + [0,0] * (max_aux - idx - 1))
        edge_labels.append(
            [0, 0, 0] + [0,0] * idx + [4,0] + [0,0] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length])
    edge_labels=torch.from_numpy(edge_labels)
    return edge_labels

def prepare_EC_info(ins_info, onto_info, device):
    instance_info=dict()
    instance_info["node_num"]=ins_info['node_num']
    instance_info["rel_num"]=ins_info['rel_num']
    instance_info["max_n"]=ins_info['max_n']
    ontology_info=dict()
    ontology_info["node_num"]=onto_info['node_num']
    ontology_info["rel_num"]=onto_info['rel_num']
    ontology_info["max_n"]=onto_info['max_n']
    return instance_info,ontology_info
'''
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
'''
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
        return sparse_mx
'''
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def no_weighted_adj(total_ent_num, triple_list):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj

def gen_adj(total_e_num, triples):
    adj_triples=list()
    for i,item in enumerate(triples[0]):
        if triples[2][i]==0:
            item[0]=triples[3][i]
            adj_triples.append(item)
    one_adj = no_weighted_adj(total_e_num, adj_triples)
    adj = one_adj
    return adj
'''
def gen_hadj(total_ent_num, statements):
    start = time.time()
    total_state_num=len(statements)
    row = list()
    col = list()
    for j,item in enumerate(statements):
        for p,i in enumerate(item):
            if p%2==0:
                row.append(i)
                col.append(j)

    data_len = len(row)
    data = np.ones(data_len)
    H = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_state_num))  
     
    n_edge = H.shape[1]# 超边矩阵
    # the weight of the hyperedge
    W = np.ones(n_edge) # 超边权重矩阵
    # the degree of the node
    DV = np.array(H.sum(1))  # 节点度; (12311,)
    # the degree of the hyperedge
    DE = np.array(H.sum(0))  # 超边的度; (24622,)
    
    invDE = sp.diags(np.power(DE, -1).flatten())  # DE^-1; 建立对角阵
    DV2 = sp.diags(np.power(DV, -0.5).flatten())  # DV^-1/2
    W = sp.diags(W)  # 超边权重矩阵
    HT = H.transpose()

    G = DV2 * H * W * invDE * HT * DV2

    logger.info('generating G costs time: {:.4f}s'.format(time.time() - start))    
    return sparse_to_tuple(G)

def prepare_adj_info(ins_info, onto_info, device):
    ins_adj = gen_hadj(ins_info['node_num'], ins_info['all_fact_ids'])
    ins_adj_info=dict()
    ins_adj_info['indices']=torch.tensor(ins_adj[0]).t().to(device)
    ins_adj_info['values']=torch.tensor(ins_adj[1]).to(device)
    ins_adj_info['size']=torch.tensor(ins_adj[2]).to(device)
    onto_adj = gen_hadj(onto_info['node_num'], onto_info['all_fact_ids'])
    onto_adj_info=dict()
    onto_adj_info['indices']=torch.tensor(onto_adj[0]).t().to(device)
    onto_adj_info['values']=torch.tensor(onto_adj[1]).to(device)
    onto_adj_info['size']=torch.tensor(onto_adj[2]).to(device)   
    return ins_adj_info, onto_adj_info