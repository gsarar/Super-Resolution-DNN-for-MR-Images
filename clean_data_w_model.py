import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import os
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from utils import *
import pdb
import shutil
import pickle
from origModel import Net
from inference_utils import *
from math import log10
from unet import UNet
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='Reading DICOM and saving as hdf5 file')
parser.add_argument('--test-data','--t', help='input hdf5 file')
parser.add_argument('--outputFile','--o', help='output hdf5 file')
parser.add_argument('--batch-size','--b', type=int, default=16, help='batch size for inference')
parser.add_argument('--model-name','--m', default='best_model.pth.tar' ,help='saved model parameters)

global args
args = parser.parse_args()

#'datasetHFD5/blurred17.h5'
test_data = args.test_data
outputFile = args.outputFile
batch_size = args.batch_size
model_name = args.model_name

learning_rate = 0.001  
seed = 0           # Seed the random number generator for reproducibility
np.random.seed(seed)

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 2, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")
    
# Define Model
# model = Net(non_linearity='relu').to(computing_device)
model = Net(non_linearity='relu',in_channel=1,channelNums=[64,64,32,1],filterSizes=[5,3,3,3]).to(computing_device)
# model = UNet().to(computing_device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

checkpoint = torch.load(model_name)
# checkpoint = torch.load('best_unet_model.pth.tar')
begin_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_loss']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

#Get the data
test_loader = create_only_inference_loader(batch_size, seed,test_data,extras=extras)


# with torch.no_grad():
#     loss_test,test_psnr = validate(test_loader,model,computing_device,criterion) 
with torch.no_grad():
    model.eval()  
    for minibatch_count, images in enumerate(test_loader, 0):
#         print(minibatch_count)
        output=model(images.to(computing_device))
        output=output.view(output.shape[0],output.shape[2],output.shape[3]).cpu().numpy()
        if minibatch_count==0:
            outputs=output
        else:
            outputs=np.concatenate((outputs, output), axis=0)
            
# write hdf5           
writeHDF5(outputFile,'data',outputs)