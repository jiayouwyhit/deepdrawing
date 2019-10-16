import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
import numpy as np
import time
from tensorboard_logger import log_value
from metric import *    
import dgl


#
# This function can calculate the gradient norm of model's generator. 
# When some parameters grad becomes nan, it will output some logs to help debugging.
#
def calculate_gradient_norm(parameters,norm_type=2):
    total_norm = 0
    flag = False
    parameters_list = []
    for name,p in parameters:
        parameters_list.append((name,p))
        param_norm = p.grad.data.norm(norm_type)
        if torch.isnan(p.grad).any():
            flag = True
        total_norm = total_norm + param_norm.item() ** norm_type

    if flag == True:
        for (name,p) in parameters_list:
            isnan = torch.isnan(p.grad).any()
            print("Tensor Name:")
            print(name)
            if isnan:
                print("NAN")
            print(p)
            print(p.grad)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class TrainConfig():
    def __init__(self,model,dataloader,valid_dataloader,test_dataloader,opt):
        self.model = model
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.opt = opt  

     
def calculate_loss(params):
    graph = params["graph"]
    graph_len = len(graph)
    y = params["y"]
    y_pred = params["y_pred"]
    len_input = params["len_input"]

    # combine
    loss = 0
    for gnum_1 in range(graph_len):
        len_node = int(len_input[gnum_1])
        loss =loss + criterion_procrustes(y_pred[gnum_1,0:len_node,:],y[gnum_1,0:len_node,:])
    loss = loss / graph_len

    return loss
            
def construct_prediction(config,graph):
    model = config.model
    opt = config.opt
    DGL_input = opt.DGL_input
    PYG_input = opt.PYG_input
    input_size = opt.max_prev_node
    
    # graph = params["graph"]
    
    graph_len = len(graph)
    if DGL_input == True: # "GraphLSTM_dgl"
        graphlist1 = []
        graphlist2 = []
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len))   
        gnum = 0
        
        device = next(model.parameters()).device # check the device that the model is running on
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
        y_pred = model(graphlist1,graphlist2)
    elif PYG_input == True: # "GraphLSTM_pyg"
        graphlist1 = []
        graphlist2 = []
        graphlist1_dgl = []
        graphlist2_dgl = []
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len)) 
        gnum = 0

        # check the device that the model is running on
        device = next(model.parameters()).device
        accu_count = 0
        from torch_geometric.data import Data
        from torch_geometric.data import Batch
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
   
            graphlist1.append(g1_data)
            graphlist2.append(g2_data)
            y_input[gnum,:,:] = g["pos"]
            accu_count = accu_count + nodenum
            gnum = gnum + 1
        # Variable and cuda
        y = torch.from_numpy(y_input).float().to(device)
        len_input = torch.from_numpy(len_input).long().to(device)

        ### Use the trained model to predict coordinates
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
        y_pred = model(g1_batch,g1_order,g1_edge_order_mask_list,g2_batch,g2_order,g2_edge_order_mask_list,len_input)
    else: # "BiLSTM"
        device = next(model.parameters()).device # check the device that the model is running on
        x_input = np.zeros((graph_len,opt.max_num_node,input_size))
        y_input = np.zeros((graph_len,opt.max_num_node,2))
        len_input = np.zeros((graph_len))
        gnum = 0
        
        for g in graph:
            len_node = g["len"]
            len_input[gnum] = len_node
            x_input[gnum,:,:] = g["x"]
            y_input[gnum,:,:] = g["pos"]
            gnum = gnum + 1
        y = torch.from_numpy(y_input).float().to(device)

        # Use the trained model to predict coordinates
        x = torch.from_numpy(x_input).float()
        x = Variable(x).to(device)
        y_pred = model(x)

    result = {
      "y":y,
      "y_pred":y_pred,
      "len_input":len_input
    }
    return result

## Test and Evaluate
def evaluate(config,test_dataloader,valid=False):
    with torch.no_grad():
        loss = 0
        start_time = time.time()
        total_len = 0
        for i, graph in enumerate(test_dataloader):
            graph_len = len(graph)
            total_len = total_len + graph_len
            # predict_params = {
            #     "graph":graph
            # }
            result = construct_prediction(config,graph)
            test_loss_params = {
                "graph":graph,
                "y":result["y"],
                "y_pred":result["y_pred"],
                "len_input":result["len_input"],
                "loss_mode":config.opt.test_loss_mode,
                "valid":valid
            }
            content_loss = calculate_loss(test_loss_params) * graph_len

            loss = loss + content_loss
            
            # limit the testing graph number, added by Yong
            if total_len > config.opt.evaluate_graph_num_limit:
                print('total testing/validating graph number: ', total_len)
                break
        if total_len == 0:
            loss_value = 0
        else:
            loss = loss / total_len
            loss_value = loss.item()
        end_time = time.time()
        duration = end_time - start_time
        return loss_value,duration


# Main Train Loop Function 
def train_model(model,dataloader,valid_dataloader,test_dataloader,opt):
    # Set model to the train mode.
    model.train()

    # Optimizers
    optimizer_G = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler_G = MultiStepLR(optimizer_G, milestones=opt.milestones, gamma=opt.lr_rate)

    config = TrainConfig(model,dataloader,valid_dataloader,test_dataloader,opt)
    
    # Initalization
    step = 0
    batch_len = len(dataloader) 
    
    # Training
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()  # timing
        # Epoch starts
        for i, graph in enumerate(dataloader):
            optimizer_G.zero_grad()
            loss = 0
            start_time = time.time()
            result = construct_prediction(config,graph)
            train_loss_params = {
                "graph":graph,
                "y":result["y"],
                "y_pred":result["y_pred"],
                "len_input":result["len_input"],
                "loss_mode":opt.train_loss_mode,
                "valid":True
            }
            loss = calculate_loss(train_loss_params) # loss function calculation
            loss.backward()
            gradient_norm = calculate_gradient_norm(model.named_parameters())

            optimizer_G.step()
            end_time = time.time()
            duration = end_time - start_time
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [Gradient Norm:%f] [Duration: %f]"
                % (epoch, opt.n_epochs, i, batch_len, loss.item(),gradient_norm, duration)
            )
            step = step + 1
            log_value('training_loss', loss, step)
        epoch_end = time.time()    
        
        # Save Model
        if epoch % opt.save_model_epoch == 0:
            model_save_path = opt.model_save_folder+'model_' + str(opt.executename) + '_' + str(epoch) + '.pkl'
            print("Epoch duration: " + str(epoch_end-epoch_start) + " Model Save Path:"+model_save_path)
            torch.save(model, model_save_path)
        
        # Validation and testing
        valid_loss, valid_duration = evaluate(config,valid_dataloader,True) 
        print(
            "Valid: [Epoch %d/%d] [G valid loss: %f] [Duration: %f]"
            % (epoch, opt.n_epochs, valid_loss, valid_duration)
         )
        log_value('validation_loss', valid_loss, epoch)  
        test_loss, test_duration = evaluate(config,test_dataloader,False) 
        print(
            "Test: [Epoch %d/%d] [G test loss: %f] [Duration: %f]"
            % (epoch, opt.n_epochs, test_loss, test_duration)
         )
        log_value('testing_loss', test_loss, epoch) 
        





##############################################################################################################
### training functions for distributed training by using Torch.distributed
##############################################################################################################

# average the gradients, which is used in the distributed training
def average_gradients(model, group):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= size

def train_model_distributed_thread(model,train_dataloader,valid_dataloader,opt, rank=0):
    # Set model to the train mode.
    model.train()
    # Optimizers
    optimizer_G = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler_G = MultiStepLR(optimizer_G, milestones=opt.milestones, gamma=opt.lr_rate)
    config = TrainConfig(model,train_dataloader,valid_dataloader,None,opt)
    
    # Initalization
    # Total Batch Len
    batch_len = len(train_dataloader) 

    # create the group
    world_size = dist.get_world_size()
    thread_list = [i for i in range(world_size)]
    group = dist.new_group(thread_list) 
    
    #  Training
    if rank == 0: # only log when rank = 0
        step = 0 # training steps
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()  # timing
        # Epoch starts
        for i, graph in enumerate(train_dataloader):
            optimizer_G.zero_grad()
            loss = 0
            start_time = time.time()
            # predict_params = {
            #     "graph":graph
            # }
            result = construct_prediction(config,graph)
            train_loss_params = {
                "graph":graph,
                "y":result["y"],
                "y_pred":result["y_pred"],
                "len_input":result["len_input"],
                "loss_mode":opt.train_loss_mode,
                "valid":True
            }
            loss = calculate_loss(train_loss_params) # loss function calculation
            
            loss.backward()
            if opt.gradient_clipping == True:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip_norm)
            gradient_norm = calculate_gradient_norm(model.named_parameters())
            average_gradients(model, group) #IMPORTANT: average the gradient from all the threads in this group
            
            if rank == 0:
                print("\n")
            
            optimizer_G.step()
            end_time = time.time()
            duration = end_time - start_time
            
            print(
                "[Rank: %d] [Epoch %d/%d] [Batch %d/%d] [G loss: %f] [Gradient Norm:%f] [Duration: %f]"
                % (rank, epoch, opt.n_epochs, i, batch_len, loss.item(),gradient_norm, duration)
            )
            
            if rank == 0: # only record the training loss of the first process
                step = step + 1
                log_value('training_loss', loss, step)
            
        epoch_end = time.time()    
        if rank == 0:
            print("Epoch duration:"+str(epoch_end-epoch_start))
        
        # Save Model
        if epoch % opt.save_model_epoch == 0 and rank == 0: # save model only in the first thread
            model_save_path = opt.model_save_folder+'model_'+str(opt.executename)+'_'+str(epoch)+'.pkl'
            print("Epoch duration:"+str(epoch_end-epoch_start)+"  Model Save Path:"+model_save_path)
            torch.save(model, model_save_path)

        if rank == 0:
            # Validation and testing
            valid_loss, valid_duration = evaluate(config,valid_dataloader,True)

            print(
                "Valid: [Epoch %d/%d] [G valid loss: %f] [Duration: %f]"
                % (epoch, opt.n_epochs, valid_loss, valid_duration)
             )
            log_value('validation_loss', valid_loss, epoch)  


def run_one_thread(rank, size, opt):
    """ Distributed Synchronous training of one thread """
    # save model and training process only for the first process
    if rank == 0: 
        if not os.path.exists(opt.model_save_folder):
            os.mkdir(opt.model_save_folder)
        ctime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        if opt.clean_tensorboard:
            if os.path.isdir("tensorboard"):
                shutil.rmtree("tensorboard")
        configure("tensorboard/"+opt.executename+"_"+ctime, flush_secs=5)
    
    # multiple GPUs
    dev_num = torch.cuda.device_count()
    device = torch.device("cuda:{}".format(rank % dev_num))
    
    # parameter initialization
    input_size = opt.max_prev_node
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    num_classes = 2
    
    # Model Initialize
    model = None
    if opt.model_select == "GraphLSTM_pyg":
        model = GraphLSTM_pyg(x_size=input_size,h_size=hidden_size,output_size=num_classes, max_node_num=opt.max_num_node)
        model = model.to(device)
    elif opt.model_select == "GraphLSTM_dgl":
        model = GraphLSTM_dgl(x_size=input_size,h_size=hidden_size,output_size=num_classes, max_node_num=opt.max_num_node)
        model = model.to(device)
    else:
        print('Guys, you select the wrong model!!!')
        return
    
    model.train()
    if rank == 0:
        print(model)
    
    # dataset
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(opt)
    
    # training
    total_train_graph_dataset,valid_graph_dataset,test_graph_dataset = get_graph_datasets(opt)
    num_train = len(total_train_graph_dataset)
    num_valid = len(valid_graph_dataset)
    num_test = len(test_graph_dataset)
    num_total = num_train + num_valid + num_test
    print("Params: %d" %calculateParamsNum(model))
    print("Train Data: %d Valid Data: %d Test Data: %d Total: %d" %
          (num_train,num_valid,num_test,num_total))
    
    valid_dataloader = None
    if rank == 0: # Do validation only when rank == 0
        valid_dataloader = DataLoader(valid_graph_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=opt.collate_fn)
    train_model_distributed_thread(model,train_set,valid_dataloader,opt, rank)


# def init_processes(rank, size, fn, backend='gloo'): # error
def init_processes(rank, size, fn, opt, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502' 
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, opt)  

    
# We use torch.distributed to train the model
def execute_train_distributed(opt): 
    size = opt.distributed_thread_size # the number of threads or GPUs we used
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run_one_thread, opt))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()   


