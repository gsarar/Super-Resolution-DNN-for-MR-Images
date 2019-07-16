import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import os
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from utils import *
import pdb
import shutil
import pickle
import torch.nn as nn
from modelSR import *
from math import log10
from train_utils import *
from unet import UNet
import argparse
import pytorch_ssim

parser = argparse.ArgumentParser(description='Parameters for model training')
parser.add_argument('--best-model-name','--be', default = 'best_model.pth.tar',help='name of the best saved model')
parser.add_argument('--save-model-name','--sa', default = 'model.pth.tar',help='name of the saved model')
parser.add_argument('--early-stopping','--e', default = 1,type=int ,help='if there is going to be early stopping or not')
parser.add_argument('--patience','--p', default = 5,type=int ,help='patience for early stopping')
parser.add_argument('--num-epochs','--n', default = 100, type=int ,help='number of training epochs')
parser.add_argument('--batch-size','--b', default = 16, type=int ,help='batch size')
parser.add_argument('--learning-rate','--lr', default = 0.001, type=float ,help='learning rate for adam optimizer')
parser.add_argument('--p-train','--pt', default = 0.88, type=float ,help='p_train is the training set ratio (p_train+p_val=1)')
parser.add_argument('--seed','--s', default = 0 ,type=int ,help='seed for reproducibility')
parser.add_argument('--pickle-name','--pi', default = 'results.pkl', help='pickle file name for saving training results')
parser.add_argument('--test-pickle-name','--tpi', default = 'test_results.pkl', help='pickle file name for saving test results')
parser.add_argument('--loss-to-min','--l', default = 'MSE', help='loss to minimize')


global args
args = parser.parse_args()

num_epochs = args.num_epochs           # Number of full passes through the dataset
batch_size = args.batch_size           # Number of samples in each minibatch
learning_rate = args.learning_rate  
p_train=args.p_train                   # percentage of training data
p_val=1- p_train                       # percentage of validation data
seed = args.seed                       # Seed the random number generator for reproducibility
save_model_name=args.save_model_name
best_model_name=args.best_model_name    
earlyStopping=args.early_stopping       # if early stopping is going to be applied or not
patience=args.patience                 #number of epochs to stop, after which there isn't decrease in validation loss,
                                        # validation loss checking is done in every 80 minibatch iterations for model saving but 
                                        # earlystopping criteria is checked at the end of each epoch in order to avoid   
                                        # oscillations
pickle_name=args.pickle_name
test_pickle_name=args.test_pickle_name  

lossToMin =args.loss_to_min
print(lossToMin)

# Setup: initialize the hyperparameters/variables    
total_loss = []
avg_minibatch_loss = []
loss_train_list = []
loss_val_list = []
loss_test_list = []
best_loss = 1000
best_epoch_loss = 1000      
np.random.seed(seed)

#transform = transforms.Compose([transforms.Resize([512,512]),transforms.RandomRotation([-180,180]),transforms.RandomHorizontalFlip(0.5), 
#                                transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, 
#                                                              transform=transform, 
#                                                              p_train=0.88, p_val=0.12,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
# Begin training procedure 
# sub-pixel convolutional neural network
model = Net(non_linearity='relu',in_channel=1,channelNums=[64,64,32,1],filterSizes=[5,3,3,3]).to(computing_device)
# U-net
# model = UNet(padding=True).to(computing_device)


if lossToMin == "SSIM":
    ## Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    criterion = pytorch_ssim.SSIM()
else:
    criterion = nn.MSELoss()



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

patienceCounter=0
for epoch in range(num_epochs):
    total_loss,loss_val_list,avg_minibatch_loss,best_loss=train_model(epoch,train_loader,val_loader,model,criterion,
                                                                            computing_device,optimizer,total_loss,
                                                                            avg_minibatch_loss,
                                                                            loss_train_list,
                                                                            loss_val_list,
                                                                            loss_test_list,
                                                                            best_loss,
                                                                            num_epochs,
                                                                            batch_size,
                                                                            learning_rate, 
                                                                            seed,
                                                                            save_model_name,
                                                                            best_model_name,lossToMin)
    
    with torch.no_grad():
        loss_val,val_psnr = validate(val_loader,model,computing_device,criterion,lossToMin) 

    #check if loss didn't increase N times. N=patience. Consecutive increase is not looked for, since it may increase oscillating
    if loss_val>best_epoch_loss:
        patienceCounter+=1   
        print("Epoch loss didn't decrease: ",patienceCounter)
    
    elif loss_val <= best_epoch_loss:
        best_epoch_loss = loss_val
        patienceCounter = 0
    
    if patienceCounter==patience:
        break
    results = {"loss_train_list":total_loss,"loss_val_list":loss_val_list,"avg_minibatch_loss":avg_minibatch_loss}
    pickle.dump( results, open(pickle_name, "wb" ) )

# testing
with torch.no_grad():
    loss_test,test_psnr = validate(test_loader,model,computing_device,criterion,lossToMin)    

loss_test_list.append(loss_test)
print('test_loss: %.7f'%(loss_test),'test_psnr: %.7f'%(test_psnr))

pickle.dump( loss_test, open( test_pickle_name, "wb" ) )
print("Training complete after", epoch, "epochs")
