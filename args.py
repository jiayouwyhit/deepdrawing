import torch

        
### program configuration
class Args():
    def __init__(self):
        ## TAG NAME
        self.executename = "copy_undefined" # You may change it
        self.main_data_folder = "./main_data_folder/"
        ##################################################################################
        ## Model Parameters
        self.hidden_size = 256   
        self.num_layers = 4 
        
        ## The number of distributed threads used, if we use torch.nn.distributed to speed up the training
        self.distributed_thread_size = 8 # the number of threads/GPUs used
        self.evaluate_graph_num_limit = 300 # the maximum number of graphs to be tested/validated during the training stage
        
        ##################################################################################
        ## Model Selection
        self.model_select_candidate = ["BiLSTM","GraphLSTM_dgl","GraphLSTM_pyg"]
        self.model_select = "GraphLSTM_pyg" # can be manually set
        
        ##################################################################################
        ######### Extra Configuration of Model Selection
        self.DGL_input = False   ### When using GraphLSTM_dgl, in the train module, the input will be switched to DGL input mode.
        self.PYG_input = False
        if self.model_select == "GraphLSTM_dgl":
            ### GraphLSTM use DGL to implement, so switch to DGL input mode.
            self.DGL_input = True
        elif self.model_select == "GraphLSTM_pyg":
            self.PYG_input = True
        #############################################################################################
        ## Training Process Basic Configuration ( can be manually set )
        self.n_epochs = 4001
        self.lr = 0.0015 # 0.001 , for Adam optimizer
        self.milestones = [50,100]  # for multistepLR
        self.lr_rate = 0.3  # for multistepLR
        self.b1 = 0.9   # for Adam optimizer
        self.b2 = 0.999 # for Adam optimizer
        self.weight_decay = 0 # for Adam optimizer
        self.gradient_clipping = False
        self.clip_norm = 100
        
        #############################################################################################
        ## Clean Tensorboard
        self.clean_tensorboard = False
        
        #############################################################################################
        ## DataLoader
        def collate_fn(batch):
            return batch
        self.collate_fn = collate_fn
        self.batch_size = 4
        self.num_workers = 0
        self.dist_train_num_workers = 2 # the number of workers for the distributed training dataloader
        
        #############################################################################################
        ## Training Process
        # self.loss_candidate = ['loss_PA']
        self.train_loss_mode = 'loss_PA' 
        self.test_loss_mode = 'loss_PA'  
        
        ## Other parameters
        self.save_model_epoch = 10   # Every save_model_epoch will save the model.
        
        ### Model Save Folder
        self.model_save_folder = self.main_data_folder + "model_save/"
        
        ### Init Graph Type
        self.set_graphtype('grid_v1')
        
        
    def set_graphtype(self,graph_type):
        ### Configure Graph Type
        self.graph_type = graph_type
        self.scale = None
        self.max_num_node = None
        self.max_prev_node = None
        self.max_num_edge = None
        self.data_path = self.main_data_folder+'data/'
        path_params = self.getDefaultPath(self.graph_type,self.data_path)
        if self.graph_type == 'grid_v1':
            self.setPath(path_params)
            self.max_num_node = 576
            self.max_prev_node = 49
            self.scale = 784.0
            self.feature_size = 1
            self.max_num_edge = 1104
        elif self.graph_type == 'grid_v2':
            self.setPath(path_params)
            self.max_num_node = 576
            self.max_prev_node = 35
            self.scale = 784.0
            self.feature_size = 1
            self.max_num_edge = 1104
        else:
            print("error in graph type : "+self.graph_type)
    def getDefaultPath(self,graph_type,data_path):
        params = {
            "target_train_dataset_file_folder" : data_path+graph_type+'_train_dataset_folder_preprocess/',
            "target_valid_dataset_file_folder" : data_path+graph_type+'_valid_dataset_folder_preprocess/',
            "target_test_dataset_file_folder" : data_path+graph_type+'_test_dataset_folder_preprocess/' 
        }
        return params
    def setPath(self,params):
        self.target_train_dataset_file_folder = params["target_train_dataset_file_folder"]
        self.target_valid_dataset_file_folder = params["target_valid_dataset_file_folder"]
        self.target_test_dataset_file_folder = params["target_test_dataset_file_folder"]