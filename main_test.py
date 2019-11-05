from test import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

### Initialize
opt = Args()
opt.set_graphtype('pivotMds_grid')
#opt.scale = 400
print(opt.__dict__)  

### Modify the epoch number to the suitable value
epoch=1500
# model_execute = "ModelName_DatasetName_TrialID"
# model_execute = "GraphLSTM_pyg-grid_v2-demo1"
model_execute = "GraphLSTM_pyg-pivotMds_grid-demo1"

model_name = 'model_'+model_execute+'_'+str(epoch)+'.pkl'
model_file = opt.main_data_folder + 'model_save/' + model_name
opt.DGL_input = False # Set this value as True, if the trained model is GraphLSTM_dgl; Otherwise, set it as False.
opt.PYG_input = True  # Set this value as False, if the trained model is GraphLSTM_pyg; Otherwise, set it as False.

folder_prefix = opt.main_data_folder + 'testing_results/'

bfs_order = False
scale_constant = 1
pred_scale_constant = 1
PA_corrected = True
cpu_mode = False
Scale_corrected = True


### Clean Folder
if os.path.exists(folder_prefix):
    shutil.rmtree(folder_prefix)
if not os.path.exists(folder_prefix):
    os.mkdir(folder_prefix)

### Read Model
if cpu_mode == True:
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
else:
    model = torch.load(model_file)

# ## Copy model in testing_results  
# shutil.copyfile(model_file,folder_prefix+model_name)

## Initialize Dataset
graph_dataset,valid_graph_dataset,test_graph_dataset = getdataset(opt)


## Begin test samples from test dataset
model_testdataset_inference_params = {
    "max_samples":24,
    "dataset":test_graph_dataset,

    "test_params":{
        "folder":folder_prefix+'test_random/',
        "model":model,
        "opt":opt,
        "cpu_mode":cpu_mode,
        "bfs_order":bfs_order,
        "Scale_corrected":Scale_corrected,
        "PA_corrected":PA_corrected,
        "scale_constant":scale_constant,
        "pred_scale_constant":pred_scale_constant
    }
}
model_inference(model_testdataset_inference_params)
