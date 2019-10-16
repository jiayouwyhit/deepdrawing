import os
import numpy as np
from time import gmtime, strftime, localtime
import random
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import Process
from torch.utils.data import Dataset, DataLoader
from tensorboard_logger import configure, log_value

from data import *
from models import *
from train import *
from args import *
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
def precheck(opt):
    opt.DGL_input = False   ### When using GraphLSTM_dgl, in the train module, the input will be switched to DGL input mode.
    opt.PYG_input = False
    if opt.model_select == "GraphLSTM_dgl":
        ### GraphLSTM use DGL to implement, so switch to DGL input mode.
        opt.DGL_input = True
    elif opt.model_select == "GraphLSTM_pyg":
        opt.PYG_input = True

    if not os.path.exists(opt.model_save_folder):
        os.mkdir(opt.model_save_folder)
    ctime = strftime("%Y-%m-%d %H:%M:%S", localtime()) # local time
    if opt.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/"+opt.executename+"_"+ctime, flush_secs=5)
    random.seed(123) 

def getdataloader(opt):
    # Dataset Init
    if opt.DGL_input == False:
        if opt.PYG_input == False:
            graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_train_dataset_file_folder)
            valid_graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_valid_dataset_file_folder)
            test_graph_dataset = Graph_sequence_from_file(dataset_file=opt.target_test_dataset_file_folder)
        else:
            graph_dataset = Graph_sequence_from_file_pyg(dataset_file=opt.target_train_dataset_file_folder)
            valid_graph_dataset = Graph_sequence_from_file_pyg(dataset_file=opt.target_valid_dataset_file_folder)
            test_graph_dataset = Graph_sequence_from_file_pyg(dataset_file=opt.target_test_dataset_file_folder)
    else:
        graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_train_dataset_file_folder)
        valid_graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_valid_dataset_file_folder)
        test_graph_dataset = Graph_sequence_from_file_dgl(dataset_file=opt.target_test_dataset_file_folder)
        
    dataloader = DataLoader(graph_dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, collate_fn=opt.collate_fn)
    valid_dataloader = DataLoader(valid_graph_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, collate_fn=opt.collate_fn)
    test_dataloader = DataLoader(test_graph_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, collate_fn=opt.collate_fn)
    num_train = len(graph_dataset)
    num_valid = len(valid_graph_dataset)
    num_test = len(test_graph_dataset)
    num_total = num_train + num_valid + num_test
    
    print("Train Data: %d Valid Data: %d Test Data: %d Total: %d" %
          (num_train,num_valid,num_test,num_total))
    return dataloader, valid_dataloader, test_dataloader

def getmodel(opt):
    input_size = opt.max_prev_node
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers # Only for "BiLSTM"
    output_size = 2

    # Model Initialize
    if opt.model_select == "GraphLSTM_pyg": # The proposed model implemented with PyG Library: we used this for evaluation
        model = GraphLSTM_pyg(x_size=input_size,h_size=hidden_size,output_size=output_size,
                               max_node_num=opt.max_num_node).cuda()
    elif opt.model_select == "GraphLSTM_dgl":   # The proposed model implemented with DGL library
        model = GraphLSTM_dgl(x_size=input_size,h_size=hidden_size,output_size=output_size,
                               max_node_num=opt.max_num_node).cuda()
    elif opt.model_select == "BiLSTM":  # The baseline model
        model = BiLSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,num_classes=output_size).cuda()
    print(model)
    print("Params: %d" %calculateParamsNum(model))
    return model

    
    

