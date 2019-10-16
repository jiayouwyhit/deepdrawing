import os
import pickle as pkl
import time
import shutil
from PIL import Image
import torch

from torch.autograd import Variable
from scipy.spatial import procrustes
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np

from data import *
from models import *
from train import *
from args import *
from utils import *
from metric import *

## Concate Images
def concate(UNIT_SIZE,UNIT_SIZE_HEIGHT,images,save_file):
    TARGET_WIDTH = UNIT_SIZE *len(images)
    imagefile = []
    j = 0
    for j in range(len(images)):
        imagefile.append(Image.open(images[j])) 
    target = Image.new('RGBA', (TARGET_WIDTH, UNIT_SIZE_HEIGHT))    
    left = 0
    right = UNIT_SIZE
    for image in imagefile:  
        target.paste(image, (left, 0, right, UNIT_SIZE_HEIGHT))
        left += UNIT_SIZE 
        right += UNIT_SIZE 
    target.save(save_file)


def transform_pos(pos,canvas_left,canvas_top,canvas_right,canvas_bottom,real_scale_cof=None):
    real_canvas_width = canvas_right - canvas_left
    real_canvas_height = canvas_bottom - canvas_top
    real_canvas_scale = min(real_canvas_width,real_canvas_height)
    min_pos = np.min(pos,0)
    max_pos = np.max(pos,0)
    x_left = min_pos[0]
    x_right = max_pos[0]
    y_top = min_pos[1]
    y_bottom = max_pos[1]
    ori_width = x_right - x_left
    ori_height = y_bottom - y_top
    max_scale = max(ori_width,ori_height)
    if max_scale == 0:
        max_scale = 1
    if real_scale_cof is None:
        real_scale_cof = real_canvas_scale / max_scale
    pos[:,0] = (pos[:,0] - x_left) * real_scale_cof + canvas_left
    pos[:,1] = (pos[:,1] - y_top) * real_scale_cof + canvas_top
    return pos, real_scale_cof

## Test Model
def model_test(testconfig):
    ## Recover Variable from TestConfig
    graph = testconfig["graph"]
    i_batch = testconfig["i_batch"]
    test_params = testconfig["test_params"]
    folder = test_params["folder"]
    opt = test_params["opt"]
    model = test_params["model"]
    cpu_mode = test_params["cpu_mode"]
    bfs_order = test_params["bfs_order"]
    Scale_corrected = test_params["Scale_corrected"]
    PA_corrected = test_params["PA_corrected"]
    scale_constant = test_params["scale_constant"]
    pred_scale_constant = test_params["pred_scale_constant"]

    DGL_input = opt.DGL_input 
    PYG_input = opt.PYG_input

    model.eval()
    if cpu_mode == True:
        model.cpu()
    print(model)
    test_loss_mode = opt.test_loss_mode
    device = next(model.parameters()).device
    
    graph = [graph]
    graph_len = len(graph)
    ### Input mode selection
    
    if DGL_input == True:
        graphlist1 = []
        graphlist2 = []
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len))        
        gnum = 0
        
        # check the device that the model is running on
        device = next(model.parameters()).device
        for g in graph:
            len_node = g["len"]
            len_input[gnum] = len_node
            nodenum = g["len"]
            g1 = g["g1"]
            g1_x = Variable(torch.from_numpy(g["x"][0:nodenum,:])).float().to(device)
            g1.ndata["x"] =g1_x
            g1.edata['edge_label'] = g1.edata['edge_label'].to(device)
            g2 = g["g2"]
            g2_x = Variable(torch.from_numpy(g["x"][0:nodenum,:])).float().to(device)
            g2.ndata["x"] = g2_x
            g2.edata['edge_label'] = g2.edata['edge_label'].to(device)
            ###
            graphlist1.append(g1)
            graphlist2.append(g2)
            y_input[gnum,:,:] = g["pos"]
            gnum = gnum + 1
        ### Variable and cuda
        y = torch.from_numpy(y_input).float().to(device)
        ### Use model to predict coordinates
        start_time =time.time()
        y_pred = model(graphlist1,graphlist2)
        duration = time.time() - start_time
    elif PYG_input == True:
        graphlist1 = []
        graphlist2 = []
        graphlist1_dgl = []
        graphlist2_dgl = []
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len))             ### Nodenum
        gnum = 0

        # check the device that the model is running on
        device = next(model.parameters()).device
        accu_count = 0
        for g in graph:
            len_node = g["len"]
            len_input[gnum] = len_node
            nodenum = g["len"]
            g_x = Variable(torch.from_numpy(g["x"][0:nodenum,:])).float()
            g1_edge_index = torch.from_numpy(g["g1_edge_index"]).long()
            g1_edge_label = torch.from_numpy(g["g1_edge_label"]).float()

            g2_edge_index = torch.from_numpy(g["g2_edge_index"]).long()
            g2_edge_label = torch.from_numpy(g["g2_edge_label"]).float()
            g1_data = Data(x=g_x,edge_index=g1_edge_index,edge_attr=g1_edge_label)#.to(device)
            g2_data = Data(x=g_x,edge_index=g2_edge_index,edge_attr=g2_edge_label)#.to(device)
            graphlist1_dgl.append(g["g1"])
            graphlist2_dgl.append(g["g2"])
            ###
            graphlist1.append(g1_data)
            graphlist2.append(g2_data)
            y_input[gnum,:,:] = g["pos"]
            accu_count = accu_count + nodenum
            gnum = gnum + 1
        ### Variable and cuda
        y = torch.from_numpy(y_input).float().to(device)
        len_input = torch.from_numpy(len_input).long().to(device)
        ### Use model to predict coordinates
        g1_batch = Batch.from_data_list(graphlist1)#.to(device)
        g2_batch = Batch.from_data_list(graphlist2)#.to(device)
        g1_dgl_batch = dgl.batch(graphlist1_dgl)
        g2_dgl_batch = dgl.batch(graphlist2_dgl)
        g1_order = dgl.topological_nodes_generator(g1_dgl_batch)
        g2_order = dgl.topological_nodes_generator(g2_dgl_batch)
        g1_order_mask = np.zeros((len(g1_order),accu_count))
        g2_order_mask = np.zeros((len(g2_order),accu_count))
        g1_edge_index = g1_batch.edge_index
        g2_edge_index = g2_batch.edge_index
        g1_edge_order_mask_list = []
        g2_edge_order_mask_list = []
        for i in range(len(g1_order)):
            order = g1_order[i]
            g1_order_mask[i,order]=1
            mask_index = g1_order_mask[i,g1_edge_index[0]]
            mask_index = np.nonzero(mask_index)    
            g1_edge_order_mask_list.append(mask_index[0])
        for i in range(len(g2_order)):
            order = g2_order[i]
            g2_order_mask[i,order]=1
            mask_index = g2_order_mask[i,g2_edge_index[0]]
            mask_index = np.nonzero(mask_index)
            g2_edge_order_mask_list.append(mask_index[0])
        g1_order = [order.to(device) for order in g1_order]
        g2_order = [order.to(device) for order in g2_order]
        g1_edge_order_mask_list = [torch.from_numpy(edge_mask).long().to(device) for edge_mask in g1_edge_order_mask_list]
        g2_edge_order_mask_list = [torch.from_numpy(edge_mask).long().to(device) for edge_mask in g2_edge_order_mask_list]
        g1_batch = g1_batch.to(device)
        g2_batch = g2_batch.to(device)
        start_time =time.time()
        y_pred = model(g1_batch,g1_order,g1_edge_order_mask_list,g2_batch,g2_order,g2_edge_order_mask_list,len_input)
        duration = time.time() - start_time
    else:
        x_input = np.zeros((graph_len,opt.max_num_node,opt.max_prev_node))
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len))             ### Nodenum
        gnum = 0
        for g in graph:
            len_node = g["len"]
            len_input[gnum] = len_node
            x_input[gnum,:,:] = g["x"]
            y_input[gnum,:,:] = g["pos"]
            gnum = gnum + 1
        ### Variable and cuda
        y = torch.from_numpy(y_input).float().to(device)


        x = torch.from_numpy(x_input).float()
        x = Variable(x).to(device)
        ### Use model to predict coordinates
        start_time = time.time()
        y_pred = model(x)
        duration = time.time() - start_time
    print("Time: %f " %(duration))
    
    ## Post Process
    y = y.reshape(opt.max_num_node,2)
    y_pred = y_pred.reshape(opt.max_num_node,2)
    y_selected = y[0:len_node,:]
    y_selected_pred = y_pred[0:len_node,:]
    y_selected = y_selected[g["x_ridx"],:]
    y_selected_pred = y_selected_pred[g["x_ridx"],:]
    
    y_selected = y_selected.data.cpu().numpy()
    y_selected_pred = y_selected_pred.data.cpu().numpy()
    ori = graph[0]["ori"]
    width = ori["width"]
    height = ori["height"]
    bounding_box = g["bounding_box"]
    scale = opt.scale
    
    if PA_corrected:
        y_selected, y_selected_pred, disparity = procrustes(y_selected, y_selected_pred)
    else:
        mtx1, mtx2, disparity = procrustes(y_selected, y_selected_pred)
    
    print(disparity)
    
    pos_ori = inv_transform_nodelist(y_selected,bounding_box,scale/scale_constant)
    pos_ori = np.array(pos_ori)
    pos_pred = inv_transform_nodelist(y_selected_pred,bounding_box,scale/pred_scale_constant) 
    pos_pred = np.array(pos_pred)

    if Scale_corrected == True:
        pos_ori, real_scale_cof = transform_pos(pos_ori,100,100,800,800)
        pos_pred,_ = transform_pos(pos_pred,100,100,900,900,real_scale_cof)
    
    ## Visualize the results.
    visualize_time = time.time()
    visualize(None,ori,"graph_"+str(i_batch)+"_real",folder,text=["graph_"+str(i_batch)+"_real"])
    visualize(pos_pred,ori,"graph_"+str(i_batch)+"_prediction",folder,mode=2,text=["graph_"+str(i_batch)+"_prediction  PA:%f" %(disparity)])
    visualize(pos_ori,ori,"graph_"+str(i_batch)+"_original",folder,mode=2,text=["graph_"+str(i_batch)+"_original"])
    images = [folder+"graph_"+str(i_batch)+"_original.png",folder+"graph_"+str(i_batch)+"_prediction.png"]
    save_file = folder+"graph_"+str(i_batch)+"_concate.png"
    concate(int(width),int(height),images,save_file)
    visualize_time = time.time() - visualize_time
    return visualize_time


### Begin Testing................
def model_inference(model_inference_params):
    max_samples = model_inference_params["max_samples"]
    graph_dataset = model_inference_params["dataset"]

    max_count = min(len(graph_dataset),max_samples)
    for count in range(max_count):
        start_time = time.time()
        
        g = graph_dataset[count]
        end_time_1 = time.time()
        testconfig = {
            "graph":g,
            "i_batch":count,
            "test_params":model_inference_params["test_params"]
        }
        vis_time = model_test(testconfig)
        end_time = time.time()
        print("%d:Total Time: %f, Total Generate Time: %f, Visualize Time: %f, Data Time: %f,  Generate Time: %f" %(count,end_time-start_time,end_time-start_time-vis_time,vis_time,end_time_1-start_time,end_time-end_time_1-vis_time))

def getdataset(opt):
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
    return graph_dataset,valid_graph_dataset,test_graph_dataset