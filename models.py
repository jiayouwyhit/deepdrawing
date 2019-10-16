import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import dgl
import inspect



#####################################################################################
#
############################# Statistics Calculation
#
#####################################################################################
def printtensor(X):
    XL = X.tolist()
    for i in range(len(XL)):
        print("line : %d "%(i))
        print(XL[i])

def calculateParamsNum(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        lk = 1
        for j in i.size():
            lk *= j
        k = k + lk
    return k



#####################################################################################
#
############################# Bi-Directional LSTM model
#
#####################################################################################
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        ## Transform Feature 
        device = next(self.parameters()).device
        self.device = device
        
        n = x.shape[0]
        h0 = torch.zeros((self.num_layers*2,n,self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_layers*2,n,self.hidden_size)).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


#####################################################################################
#
############################# GraphLSTM by using DGL library
#
#####################################################################################

"""
We assume that there are only two types of edges: 0 and 1
"""
class TimeCounter:
    def __init__(self):
        self.timelist = {}
        self.start_time = 0
    def clear(self):
        self.timelist = {}
    def start(self):
        self.start_time = time.time()
    def record(self,name):
        if name not in self.timelist:
            self.timelist[name] = 0
        self.timelist[name] = self.timelist[name] + time.time() - self.start_time
        self.start_time = time.time()
    def print_out(self):
        totaltime = 0
        
        for key,value in self.timelist.items():
            #print("%s : %f" %(key,value))
            totaltime = totaltime + value
        print("%s : %f" %("totaltime", totaltime))
        for key,value in self.timelist.items():
            print("%s : %f (%f%%)" %(key,value,value / totaltime * 100))
LocalTimeCounter_1 = TimeCounter()
LocalTimeCounter_2 = TimeCounter()


"""
- We tried to improve its efficiency
- We assume that there are only two types of edges: 0 and 1
"""
class ImprovedGraphLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ImprovedGraphLSTMCell, self).__init__()
        self.h_size = h_size
        self.x_size = x_size
        
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=False) 
        
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        
        self.U_iou = nn.Linear(2*h_size, 3*h_size, bias=False) # U_iou_t0, U_iou_t1
        self.U_f = nn.Linear(2*h_size, h_size, bias=False)   # U_f_t0, U_f_t1 


    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 'edge_label': edges.data['edge_label']}

    def reduce_func(self, nodes):
        # check the device that the model is running on
        device = next(self.parameters()).device
        self.device = device
        
        # nodes.mailbox['edge_label'].size(): (batch_size, in_degree）
        # nodes.mailbox['h'].size(): (batch_size, in_degree, h_size)
        edge_labels = nodes.mailbox['edge_label']#.to(self.device) # edge types  （batch_size, in_degree）
        edge_labels = edge_labels.repeat(1, 1, self.h_size) # mask for Type 1 edges
        edge_labels_reverse = (-1 * edge_labels) + 1 # mask for Type 0 edges
        
        h_t1 = edge_labels * nodes.mailbox['h']#.to(self.device)
        h_t0 = edge_labels_reverse * nodes.mailbox['h']#.to(self.device)
        
        h_two_type = torch.cat((h_t0, h_t1), 2) # (batch_size, in_degree, 2*h_size)
        iou_mid = torch.sum(self.U_iou(h_two_type), 1) # (batch_size, 3 * h_size)
        f_mid = self.U_f(h_two_type) # (batch_size, in_degree, h_size)
        
        in_degree = f_mid.size(1)
        f = torch.sigmoid(nodes.data['wf_x'].unsqueeze(1).repeat(1, in_degree, 1) + f_mid + self.b_f)
        
        # second term of c
        c = torch.sum(f * nodes.mailbox['c'], 1) 

        return {'iou_mid': iou_mid,  'c': c}

    def apply_node_func(self, nodes):
        device = next(self.parameters()).device
        self.device = device
    
        iou = nodes.data['iou'] + nodes.data['iou_mid'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c}  


class SingleRoundGraphLSTM_dgl(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 ):
        super(SingleRoundGraphLSTM_dgl, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        
        self.cell = ImprovedGraphLSTMCell(x_size, h_size)    
    def forward(self, g, h, c): 
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        
        device = next(self.parameters()).device
        self.device = device
        self.cell.to(self.device)
        
        # init of data
        g.ndata['iou'] = self.cell.W_iou(g.ndata['x']).float().to(self.device) 
        g.ndata['wf_x'] = self.cell.W_f(g.ndata['x']).float().to(self.device) 
        g.ndata['iou_mid'] = torch.zeros(g.number_of_nodes(), 3 * self.h_size).to(self.device) 
        g.ndata['h'] = h.to(self.device) 
        g.ndata['c'] = c.to(self.device) 

        dgl.prop_nodes_topo(g)
        
        return g
       

class GraphLSTM_dgl(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 output_size,
                 max_node_num
                ):
        super(GraphLSTM_dgl, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.output_size = output_size
        self.max_node_num = max_node_num 
        
        self.fc_output = nn.Linear(2*h_size,output_size)
        self.forwardRound = SingleRoundGraphLSTM_dgl(x_size, h_size)
        self.backwardRound = SingleRoundGraphLSTM_dgl(x_size,h_size)
        
        
    def forward(self, g1_list, g2_list):
        input_start_time = time.time()
        g1 = dgl.batch(g1_list)
        g2 = dgl.batch(g2_list)
        n = g1.number_of_nodes()
        
        # check the device that the model is running on
        device = next(self.parameters()).device
        self.device = device
        self.forwardRound.to(self.device)
        self.backwardRound.to(self.device)
        
        # forward round
        start_time = time.time()
        h1 = torch.zeros((n, self.h_size)).float().to(self.device)
        c1 = torch.zeros((n, self.h_size)).float().to(self.device)
        g1 = self.forwardRound(g1,h1,c1)
        
        # backward round
        h2 = torch.zeros((n, self.h_size)).float().to(self.device)
        c2 = torch.zeros((n, self.h_size)).float().to(self.device)
        g2 = self.backwardRound(g2,h2,c2)
        #end_time = time.time()
        g1_rlist = dgl.unbatch(g1)
        g2_rlist = dgl.unbatch(g2)
        
        h1_list = torch.zeros((len(g1_list),self.max_node_num,self.h_size)).float().to(self.device)
        h2_list = torch.zeros((len(g2_list),self.max_node_num,self.h_size)).float().to(self.device)
        batch_size = len(g1_list)
        for i in range(batch_size):
            nodenum = g1_rlist[i].number_of_nodes()
            h1_list[i,0:nodenum,:] = g1_rlist[i].ndata["h"]
            h2_list[i,0:nodenum,:] = g2_rlist[i].ndata["h"]
        h_out = torch.cat((h1_list,h2_list),dim=2)
        
        real_output = self.fc_output(h_out)
        
        return real_output





#####################################################################################
#
############################# GraphLSTM by using the PyG library
#
#####################################################################################
class MessagePassing_custom(nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing_custom, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, size=None, **kwargs):
        assert aggr in ['add', 'mean', 'max','none']
        kwargs['edge_index'] = edge_index

        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0) if size is None else size
                tmp = torch.index_select(tmp, 0, edge_index[0])
                message_args.append(tmp)
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0) if size is None else size
                tmp = torch.index_select(tmp, 0, edge_index[1])
                message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out
    
class SingleRoundGraphLSTM_pyg(MessagePassing_custom):
    def __init__(self,
                 x_size,
                 h_size,
                 max_node_num
                 ):
        super(SingleRoundGraphLSTM_pyg, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.max_node_num = max_node_num
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=False) 
        
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        
        self.U_iou = nn.Linear(2*h_size, 3*h_size, bias=False) # U_iou_t0, U_iou_t1
        self.U_f = nn.Linear(2*h_size, h_size, bias=False)   # U_f_t0, U_f_t1 

    def forward(self, g_data, g_order_index,g_order,len_input): 
        # check the device that the model is running on
        device = next(self.parameters()).device
        self.device = device
        graph_num = g_data.num_graphs
        x, edge_index, edge_label, batch = g_data.x, g_data.edge_index, g_data.edge_attr, g_data.batch
        
        n = x.shape[0]
        h = torch.zeros((n,self.h_size)).to(device)
        c = torch.zeros((n,self.h_size)).to(device)
        iou = self.W_iou(x).float().to(self.device)
        wf_x = self.W_f(x).float().to(self.device)
        iou_mid = torch.zeros(n,3*self.h_size).to(self.device)
        out = {"c":c,"iou_mid":iou_mid}
        out = self.update(out,iou,g_order_index[0],h)
        order_num = len(g_order)
        for i in range(1,order_num):
            LocalTimeCounter_1.start()
            h = out["h"]
            c = out["c"]
            iou_mid = out["iou_mid"]
            mask_index = g_order[i]
            edge_index_mask = torch.index_select(edge_index,1,mask_index)                                  
            edge_label_mask = torch.index_select(edge_label,0,mask_index)
            out = self.propagate('none', edge_index_mask , wf_x=wf_x,iou_x=iou,iou_mid=iou_mid, 
                                edge_labels=edge_label_mask, h=h,c=c, node_num=n, order_index=g_order_index[i])
        h = out["h"]
        h_output = torch.zeros((graph_num,self.max_node_num,self.h_size)).float().to(device)
        accu_count = 0
        for i in range(graph_num):
            g_num = len_input[i]
            h_output[i,0:g_num,:] =  h[accu_count:accu_count + g_num,:]
            accu_count = accu_count + g_num
        return h_output
    
    def message(self, wf_x_j, h_j, c_j, edge_index, edge_labels, node_num, iou_mid, c):
        #start_time = time.time()
        # check the device that the model is running on        
        if h_j.shape[0] == 0:
            print("ERROR h_j shape is 0")
        device = next(self.parameters()).device
        self.device = device
        # edge_labels (num_edges)
        edge_labels = edge_labels.repeat(1, self.h_size) # mask for Type 1 edges
        edge_labels_reverse = (-1 * edge_labels) + 1 # mask for Type 0 edges
        
        h_t1 = edge_labels * h_j
        h_t0 = edge_labels_reverse * h_j
        
        h_two_type = torch.cat((h_t0, h_t1), 1) # (num_edges, 2*h_size)
        iou_pre_mid = self.U_iou(h_two_type) # (num_edges, 3*h_size)
        from torch_scatter import scatter_add
        iou_mid = scatter_add(iou_pre_mid, edge_index[0], 0, iou_mid, node_num, 0)  ## (batch_size, 3*h_size)
        #iou_mid = torch.sum(self.U_iou(h_two_type), 1) # (batch_size, 3 * h_size)
        f_mid = self.U_f(h_two_type) # (num_edges, h_size)
        
        #in_degree = f_mid.size(1)
        f = torch.sigmoid(wf_x_j + f_mid + self.b_f)
        c_j = torch.clamp(c_j,min=-1e14,max=1e14)
        #f = torch.sigmoid(nodes.data['wf_x'].unsqueeze(1).repeat(1, in_degree, 1) + f_mid + self.b_f)
        f_post_mid = f * c_j
        c = scatter_add(f_post_mid, edge_index[0], 0, c, node_num, 0)
        # second term of c
        #c = torch.sum(f * nodes.mailbox['c'], 1) 
        return {'iou_mid': iou_mid,  'c': c}
        

    def update(self, aggr_out, iou_x, order_index, h):
        # check the device that the model is running on
        iou_mid = aggr_out['iou_mid']
        c = aggr_out['c']
        iou_x_select = torch.index_select(iou_x,0,order_index)
        iou_mid_select = torch.index_select(iou_mid,0,order_index)
        iou = iou_x_select + iou_mid_select + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c_select = torch.index_select(c,0,order_index)
        c_temp = i * u + c_select
        c[order_index] = c_temp
        h[order_index,:] = o * torch.tanh(c_temp)
        return {'h': h, 'c': c, 'iou_mid':iou_mid}     
            
class GraphLSTM_pyg(nn.Module):
    def __init__(self, 
             x_size,
             h_size,
             output_size,
             max_node_num):
         # define linear layers
        super(GraphLSTM_pyg, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.output_size = output_size
        self.max_node_num = max_node_num
        
        self.fc_output = nn.Linear(2*h_size,output_size)
        self.forwardRound = SingleRoundGraphLSTM_pyg(x_size, h_size, max_node_num)
        self.backwardRound = SingleRoundGraphLSTM_pyg(x_size, h_size, max_node_num)
    def forward(self, g1_data,g1_order,g1_order_mask, g2_data,g2_order,g2_order_mask,len_input):
        result_1 = self.forwardRound(g1_data,g1_order,g1_order_mask,len_input)
        result_2 = self.backwardRound(g2_data,g2_order,g2_order_mask,len_input)
        h_out = torch.cat((result_1,result_2),dim=2)
        real_output = self.fc_output(h_out)
        return real_output





class SingleRoundGraphLSTM_pyg_double(MessagePassing_custom):
    def __init__(self,
                 x_size,
                 h_size,
                 max_node_num
                 ):
        super(SingleRoundGraphLSTM_pyg_double, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.max_node_num = max_node_num
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=False) 
        
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        
        self.U_iou = nn.Linear(2*h_size, 3*h_size, bias=False) # U_iou_t0, U_iou_t1
        self.U_f = nn.Linear(2*h_size, h_size, bias=False)   # U_f_t0, U_f_t1 

    def forward(self, x, edge_index, edge_label, batch, g_order_index,g_order): 
        # check the device that the model is running on
        device = next(self.parameters()).device
        self.device = device
        
        n = x.shape[0]
        h = torch.zeros((n,self.h_size)).to(device)
        c = torch.zeros((n,self.h_size)).to(device)
        iou = self.W_iou(x).float().to(self.device)
        wf_x = self.W_f(x).float().to(self.device)
        iou_mid = torch.zeros(n,3*self.h_size).to(self.device)
        out = {"c":c,"iou_mid":iou_mid}
        out = self.update(out,iou,g_order_index[0],h)
        order_num = len(g_order)
        for i in range(1,order_num):
            LocalTimeCounter_1.start()
            h = out["h"]
            c = out["c"]
            iou_mid = out["iou_mid"]
            mask_index = g_order[i]
            edge_index_mask = torch.index_select(edge_index,1,mask_index)                                  
            edge_label_mask = torch.index_select(edge_label,0,mask_index)
            out = self.propagate('none', edge_index_mask , wf_x=wf_x,iou_x=iou,iou_mid=iou_mid, 
                                edge_labels=edge_label_mask, h=h,c=c, node_num=n, order_index=g_order_index[i])
        h = out["h"]
        return h
    
    def message(self, wf_x_j, h_j, c_j, edge_index, edge_labels, node_num, iou_mid, c):
        #start_time = time.time()
        if h_j.shape[0] == 0:
            print("ERROR h_j shape is 0")

        # check the device that the model is running on                
        device = next(self.parameters()).device
        self.device = device
        # edge_labels (num_edges)
        edge_labels = edge_labels.repeat(1, self.h_size) # mask for Type 1 edges
        edge_labels_reverse = (-1 * edge_labels) + 1 # mask for Type 0 edges
        
        h_t1 = edge_labels * h_j
        h_t0 = edge_labels_reverse * h_j
        
        h_two_type = torch.cat((h_t0, h_t1), 1) # (num_edges, 2*h_size)
        iou_pre_mid = self.U_iou(h_two_type) # (num_edges, 3*h_size)
        #from torch_scatter import scatter_mean, scatter_add
        from torch_scatter import scatter_add
        iou_mid = scatter_add(iou_pre_mid, edge_index[0], 0, iou_mid, node_num, 0)  ## (batch_size, 3*h_size)
        #iou_mid = torch.sum(self.U_iou(h_two_type), 1) # (batch_size, 3 * h_size)
        f_mid = self.U_f(h_two_type) # (num_edges, h_size)
        
        #in_degree = f_mid.size(1)
        f = torch.sigmoid(wf_x_j + f_mid + self.b_f)
        c_j = torch.clamp(c_j,min=-1e14,max=1e14)
        f_post_mid = f * c_j

        c = scatter_add(f_post_mid, edge_index[0], 0, c, node_num, 0)
        return {'iou_mid': iou_mid,  'c': c}
        

    def update(self, aggr_out, iou_x, order_index, h):
        # check the device that the model is running on
        iou_mid = aggr_out['iou_mid']
        c = aggr_out['c']
        iou_x_select = torch.index_select(iou_x,0,order_index)
        iou_mid_select = torch.index_select(iou_mid,0,order_index)
        iou = iou_x_select + iou_mid_select + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c_select = torch.index_select(c,0,order_index)
        c_temp = i * u + c_select
        c[order_index] = c_temp
        h[order_index,:] = o * torch.tanh(c_temp)
        return {'h': h, 'c': c, 'iou_mid':iou_mid}     
                
class GraphLSTM_pyg_double(nn.Module):
    def __init__(self, 
             x_size,
             h_size,
             output_size,
             max_node_num):
         # define linear layers
        super(GraphLSTM_pyg_double, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.output_size = output_size
        self.max_node_num = max_node_num 
        
        self.fc_output = nn.Linear(2*h_size,output_size)
        self.forwardRound_1 = SingleRoundGraphLSTM_pyg_double(x_size, h_size, max_node_num)
        self.forwardRound_2 = SingleRoundGraphLSTM_pyg_double(h_size, h_size, max_node_num)
        self.backwardRound_1 = SingleRoundGraphLSTM_pyg_double(x_size, h_size, max_node_num)
        self.backwardRound_2 = SingleRoundGraphLSTM_pyg_double(h_size, h_size, max_node_num)
    def forward(self, g1_data,g1_order,g1_order_mask, g2_data,g2_order,g2_order_mask,len_input):
        device = next(self.parameters()).device
        self.device = device
        result_f1 = self.forwardRound_1(g1_data.x, g1_data.edge_index, g1_data.edge_attr, g1_data.batch,g1_order,g1_order_mask)
        result_f2 = self.forwardRound_2(result_f1, g1_data.edge_index, g1_data.edge_attr, g1_data.batch,g1_order,g1_order_mask)
        graph_num = g1_data.num_graphs
        h1_output = torch.zeros((graph_num,self.max_node_num,self.h_size)).float().to(device)
        accu_count = 0
        for i in range(graph_num):
            g_num = len_input[i]
            h1_output[i,0:g_num,:] =  result_f2[accu_count:accu_count + g_num,:]
            accu_count = accu_count + g_num
        result_b1 = self.backwardRound_1(g2_data.x, g2_data.edge_index, g2_data.edge_attr, g2_data.batch,g2_order,g2_order_mask)
        result_b2 = self.backwardRound_2(result_b1, g2_data.edge_index, g2_data.edge_attr, g2_data.batch,g2_order,g2_order_mask)
        graph_num = g2_data.num_graphs
        h2_output = torch.zeros((graph_num,self.max_node_num,self.h_size)).float().to(device)
        accu_count = 0
        for i in range(graph_num):
            g_num = len_input[i]
            h2_output[i,0:g_num,:] =  result_b2[accu_count:accu_count + g_num,:]
            accu_count = accu_count + g_num
        h_out = torch.cat((h1_output,h2_output),dim=2)
        real_output = self.fc_output(h_out)
        return real_output
