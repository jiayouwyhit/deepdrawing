import os
import pickle as pkl
import math 
import networkx as nx
import numpy as np
import scipy.sparse as sp
import copy
import dgl
import torch
import torch.distributed as dist
from math import ceil
from random import Random
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from collections import defaultdict


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output



def EuclideanDistances(A):
    B = A
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return ED

def transform_ref_pos(ref_pos):
    if ref_pos is None:
        return None
    distance = EuclideanDistances(ref_pos)
    mean = np.mean(distance, axis=1)
    variance = np.var(distance, axis=1)
    new_ref_pos = np.concatenate((mean,variance), axis=1)
    return new_ref_pos
def get_distance(ref_pos):
    if ref_pos is None:
        return None
    distance = EuclideanDistances(ref_pos)
    return distance


def get_bounding_box(nodelist):
    nodenum = len(nodelist)
    pos = np.zeros((nodenum,2))
    for node in nodelist:
        id = node[0]
        cx = float(node[2])
        cy = float(node[3])
        pos[id][0] = cx
        pos[id][1] = cy
    pos2 = pos
    left = pos2[0][0]
    right = pos2[0][0]
    top = pos2[0][1]
    bottom = pos2[0][1]
    for i in range(nodenum):
        if left > pos2[i][0]:
            left = pos2[i][0]
        if right < pos2[i][0]:
            right = pos2[i][0]
        if top > pos2[i][1]:
            top = pos2[i][1]
        if bottom < pos2[i][1]:
            bottom = pos2[i][1]
    bounding_box = {"left":left,"right":right,"top":top,"bottom":bottom}
    return pos2,bounding_box
def transform_nodelist(pos2,bounding_box,scale):
    nodenum, _ = np.shape(pos2)
    left = bounding_box["left"]
    top = bounding_box["top"]
    new_w = scale
    new_h = scale
    for i in range(nodenum):
        if new_w == 0:
            pos2[i][0] = 0
        else:
            pos2[i][0] = float((pos2[i][0] - left)) / float(new_w)
        if new_h == 0:
            pos2[i][1] = 0
        else:
            pos2[i][1] = float((pos2[i][1] - top)) / float(new_h)
    return pos2

def inv_transform_nodelist(pos,bounding_box,scale): 
    m, n = np.shape(pos)
    pos_left = pos[0][0]
    pos_right = pos[0][0]
    pos_bottom = pos[0][1]
    pos_top = pos[0][1]
    for i in range(m):
        if pos_left>pos[i][0]:
            pos_left = pos[i][0]
        if pos_top>pos[i][1]:
            pos_top = pos[i][1]
        if pos_right<pos[i][0]:
            pos_right = pos[i][0]
        if pos_bottom<pos[i][1]:
            pos_bottom = pos[i][1]
    
    for i in range(m):
        pos[i][0] = (pos[i][0] - pos_left)
        pos[i][1] = (pos[i][1] - pos_top)

    left = bounding_box["left"]
    top = bounding_box["top"]
    new_w = scale
    new_h = scale
    for i in range(m):
        pos[i][0] = (pos[i][0])*new_w + left
        pos[i][1] = (pos[i][1])*new_h + top
        
    return pos


def calc_max_num_node(graphlist,dataset_file):
    max_num_node = 0
    for idx in range(len(graphlist)):
        pathname = graphlist[idx]
        object1 = {}
        with open(dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        nodelist = object1["nodelist"]
        max_num_node = max([max_num_node,len(nodelist)])
    return max_num_node
def calc_max_num_edge(graphlist,dataset_file):
    max_num_edge = 0
    for idx in range(len(graphlist)):
        pathname = graphlist[idx]
        object1 = {}
        with open(dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        linelist = object1["linelist"]
        max_num_edge = max([max_num_edge,len(linelist)])
    return max_num_edge  
def calc_max_prev_node(graphlist,dataset_file, iter=20000,topk=10):
    max_prev_node = []
    for i in range(iter):
        if i % (iter / 5) == 0:
            print('iter {} times'.format(i))
        idx = np.random.randint(len(graphlist))
        pathname = graphlist[idx]
        object1 = {}
        with open(dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        linelist = object1["linelist"]
        graph = defaultdict(list)
        for line in linelist:
            graph[line[0]].append(line[1])
            graph[line[1]].append(line[0])

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
        adj_copy = adj.copy()
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
        max_prev_node.append(max_encoded_len)
    max_prev_node = sorted(max_prev_node)[-1*topk:]
    return max_prev_node
def calc_scale(graphlist,dataset_file):
    maxwidth = 0
    maxheight = 0
    for idx in range(len(graphlist)):
        pathname = graphlist[idx]
        object1 = {}
        with open(dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        nodelist = object1["nodelist"]
        width = object1["width"]
        height = object1["height"]
        _,bounding_box = get_bounding_box(nodelist)
        newwidth = bounding_box["right"]-bounding_box["left"]
        newheight = bounding_box["bottom"]-bounding_box["top"]
        maxwidth = max([maxwidth,newwidth])
        maxheight = max([maxheight,newheight])
    return max([maxwidth,maxheight])


########## Graph_sequence_from_file
class Graph_sequence_from_file_dgl(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.graphlist = os.listdir(self.dataset_file)
        
    def __len__(self):
        return len(self.graphlist)
    
    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        object1 = {}
        with open(self.dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        nodenum = object1["len"]
        graph = object1["graph"]
        g1 = dgl.DGLGraph()
        g2 = dgl.DGLGraph()
        # add nodes into the graph; nodes are labeled from 0 to (nodenum - 1)
        g1.add_nodes(nodenum)
        g2.add_nodes(nodenum)
        # real edges
        for i in range(nodenum):
            for j in range(len(graph[i])):
                tgt = graph[i][j]
                src = i
                if src<tgt:
                    g1.add_edges(src,tgt)
                if src>tgt:
                    g2.add_edges(src,tgt)
        real_edge_num = g1.number_of_edges()
        
        # fake edges due to BFS order
        for i in range(nodenum-1):
            g1.add_edges(i,i+1)
            g2.add_edges(nodenum-1-i,nodenum-2-i)
        all_edge_num = g1.number_of_edges()
        
        # initialize all the node and edge features
        g1.set_n_initializer(dgl.init.zero_initializer)
        g2.set_n_initializer(dgl.init.zero_initializer)
        g1.set_e_initializer(dgl.init.zero_initializer)
        g2.set_e_initializer(dgl.init.zero_initializer)
        
        # add label to edges
        g1.edata['edge_label'] = torch.ones(all_edge_num, 1)
        g1.edata['edge_label'][0:real_edge_num] = 0
        g2.edata['edge_label'] = torch.ones(all_edge_num, 1)
        g2.edata['edge_label'][0:real_edge_num] = 0
        
        object1["g1"] = g1
        object1["g2"] = g2
        return object1


class Graph_sequence_from_file_pyg(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.graphlist = os.listdir(self.dataset_file)
        
    def __len__(self):
        return len(self.graphlist)
    
    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        object1 = {}
        with open(self.dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        nodenum = object1["len"]
        graph = object1["graph"]
        g1_edge_index = []
        g2_edge_index = []

        # add nodes into the graph; nodes are labeled from 0 to (nodenum - 1)
        g1 = dgl.DGLGraph()
        g2 = dgl.DGLGraph()

        # add nodes into the graph; nodes are labeled from 0 to (nodenum - 1)
        g1.add_nodes(nodenum)
        g2.add_nodes(nodenum)
        # real edges
        for i in range(nodenum):
            for j in range(len(graph[i])):
                tgt = graph[i][j]
                src = i
                if src<tgt:
                    g1_edge_index.append([tgt,src])
                    g1.add_edges(src,tgt)
                if src>tgt:
                    g2_edge_index.append([tgt,src])
                    g2.add_edges(src,tgt)
        real_edge_num = len(g1_edge_index)

        # fake edges due to BFS order
        for i in range(nodenum-1):
            g1_edge_index.append([i+1,i])
            g1.add_edges(i,i+1)
            g2_edge_index.append([nodenum-2-i,nodenum-1-i])
            g2.add_edges(nodenum-1-i,nodenum-2-i)
        all_edge_num = len(g1_edge_index)
        g1_edge_index = np.asarray(g1_edge_index).T
        g2_edge_index = np.asarray(g2_edge_index).T

        # add label to edges
        g1_edge_label = np.ones((all_edge_num, 1))
        g1_edge_label[0:real_edge_num] = 0
        g2_edge_label = np.ones((all_edge_num, 1))
        g2_edge_label[0:real_edge_num] = 0

        object1["g1_edge_index"] = g1_edge_index
        object1["g2_edge_index"] = g2_edge_index
        object1["g1_edge_label"] = g1_edge_label
        object1["g2_edge_label"] = g2_edge_label
        object1["g1"]=g1
        object1["g2"]=g2
        return object1

########## Graph_sequence_from_file
class Graph_sequence_from_file(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.graphlist = os.listdir(self.dataset_file)
        
    def __len__(self):
        return len(self.graphlist)
    
    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        object1 = {}
        with open(self.dataset_file + pathname,"rb") as f:
            object1 = pkl.load(f)
        return object1
    
    
    

############# Fetching Datasets for Distributed Training######################################
# get the datasets
def get_graph_datasets(opt):
    train_graph_dataset = None
    valid_graph_dataset = None
    test_graph_dataset = None

    if opt.DGL_input == False:
        train_graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_train_dataset_file_folder)
        valid_graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_valid_dataset_file_folder)
        test_graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_test_dataset_file_folder)
    else: 
        train_graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_train_dataset_file_folder)
        valid_graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_valid_dataset_file_folder)
        test_graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_test_dataset_file_folder)   

    return train_graph_dataset,valid_graph_dataset,test_graph_dataset




# partition the datasets
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(opt):
    training_graph_dataset,_,_ = get_graph_datasets(opt)
    
    size = dist.get_world_size()
    overall_bsz = opt.batch_size
    bsz = overall_bsz / float(size)
    bsz = int(bsz) 

    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(training_graph_dataset, partition_sizes) # Divide the data into size parts equally
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, collate_fn=opt.collate_fn,batch_size=bsz, num_workers=opt.dist_train_num_workers, shuffle=True)
    
    return train_set, bsz
    
############# Fetching Datasets for Distributed Training######################################