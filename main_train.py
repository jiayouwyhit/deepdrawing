import os
import torch
from pre_train import *


#Parameter Initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
opt = Args()

opt.set_graphtype('grid_v2')
# opt.executename = "ModelName_DatasetName_TrialID"
opt.executename = "GraphLSTM_pyg-grid_v2-demo1"

# Set the chosen model and related parameters
# All the model candidates:
#   "BiLSTM": a 4-layer Bidirectional-LSTM model
#   "GraphLSTM_dgl":  the proposed Graph LSTM model implemented with the DGL library
#   "GraphLSTM_pyg": the proposed Graph LSTM model implemented with the PyG library
opt.model_select = "GraphLSTM_pyg" 
precheck(opt)
print(opt.__dict__)

# Start to train the model
model = getmodel(opt)
dataloader, valid_dataloader, test_dataloader = getdataloader(opt)
train_model(model,dataloader, valid_dataloader, test_dataloader,opt)