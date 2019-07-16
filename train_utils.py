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
from math import log10
import torch.nn as nn

class KneeDataset(Dataset):
    """Custom Dataset class for the Knee Dataset.
    """
    
    def __init__(self, inputData,gtData,transform=transforms.ToTensor(), color='L'):
        """
        Args:
        -----
        - transform: A torchvision.transforms object - 
                     transformations to apply to each image
                     (Can be "transforms.Compose([transforms])")
        - color: Specifies image-color format to convert to 
                 (default is L: 8-bit pixels, black and white)

        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset
        """
        key='data'
        self.transform = transform
        self.color = color
        self.image_dir = "/home/ubuntu/gsarar/subtle_medical_transfer"
        
        self.inputData= readHDF5(os.path.join(self.image_dir,inputData),key)
#         print('here')

        self.gtData= readHDF5(os.path.join(self.image_dir,gtData),key)
#         print('here')

        
    def __len__(self):
        
        # Return the total number of data samples
        return self.inputData.shape[0]


    def __getitem__(self, ind):
        """Returns the image and its ground truth at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (inputImage, groundTruth)
        """
        image=self.inputData[ind,:,:]
        groundTruth=self.gtData[ind,:,:]
        
        # If a transform is specified, apply it
        if self.transform is not None:
#             print(self.transform)
#             print(image.shape)
            image=Image.fromarray(image)
            image = self.transform(image)
            groundTruth=Image.fromarray(groundTruth)
            groundTruth = self.transform(groundTruth)
            
        # Verify that image is in Tensor format
        if type(image) is not torch.Tensor:
            image=Image.fromarray(image)
            image = transform.ToTensor(image)
        if type(groundTruth) is not torch.Tensor:
            groundTruth = Image.fromarray(groundTruth)
            groundTruth = transform.ToTensor(groundTruth)
            
        # Return the image and its label
        return (image, groundTruth)    

def create_split_loaders(batch_size, seed, transform=transforms.ToTensor(),
                         p_train=0.88, p_val=0.12, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Get create a Dataset object
    dataset = KneeDataset('blurredTrain.h5','cleanTrain.h5',transform)
    testDataset = KneeDataset('blurredTest.h5','cleanTest.h5',transform)
#     dataset = KneeDataset('datasetHFD5/blurred11.h5','datasetHFD5/clean11.h5',transform)
#     testDataset = KneeDataset('datasetHFD5/blurred13.h5','datasetHFD5/clean13.h5',transform)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

#     print('here')
    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    sample_test = SubsetRandomSampler(list(range(len(testDataset))))

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)
    
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(testDataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)


    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)

def validate(loader,model,computing_device,criterion,lossToMin):
    N_minibatch_loss = 0.0   
    total_psnr=0.0

    # switch to evaluate mode
    model.eval()  
    for minibatch_count, (images, gt) in enumerate(loader, 0):
        #print(minibatch_count)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, gt = images.to(computing_device), gt.to(computing_device)
        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        
        if lossToMin=="SSIM":
            loss = -criterion(outputs, gt)
        else:
            loss = criterion(outputs, gt)
            psnr = 10 * log10(1 / loss.item())
            total_psnr += psnr 
        
        N_minibatch_loss += loss
        
    N_minibatch_loss /= (minibatch_count+1)
    total_psnr /= (minibatch_count+1)

    return N_minibatch_loss,total_psnr

def save_checkpoint(state, is_best, filename='model.pth.tar',bestName='best_model.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestName)


def train_model(epoch,train_loader,val_loader,model,criterion,computing_device,optimizer,total_loss,
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
                                                                            best_model_name,lossToMin):
 
    
    N = 5
    N_minibatch_loss = 0.0   
    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, gt) in enumerate(train_loader, 0):
        model.train(True)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, gt = images.to(computing_device), gt.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        
        if lossToMin=="SSIM":
            loss = -criterion(outputs, gt)
        else:
            loss = criterion(outputs, gt)
            

        # Compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        #Updating the loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss
        if minibatch_count!=0 and minibatch_count % N == 0:    
            
            # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.7f' %(epoch + 1, minibatch_count, N_minibatch_loss))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0
        
        if minibatch_count!=0 and minibatch_count % (12*N) == 0:        
            #cross-validation
            with torch.no_grad():
                loss_val,val_psnr = validate(val_loader,model,computing_device,criterion,lossToMin) 

            loss_val_list.append(loss_val)
            print('val_loss: %.7f'%(loss_val),'val_psnr: %.7f'%(val_psnr))
            
            # remember best loss and save checkpoint
            is_best = loss_val <= best_loss
            best_loss = min(loss_val, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best,save_model_name,best_model_name)

            
    print("Finished", epoch + 1, "epochs of training")
    return total_loss, loss_val_list,avg_minibatch_loss,best_loss      
        
class KneeDatasetInference(Dataset):
    """Custom Dataset class for the Knee Dataset.
    """
    
    def __init__(self, inputData, color='L'):
        """
        Args:
        -----
        - transform: A torchvision.transforms object - 
                     transformations to apply to each image
                     (Can be "transforms.Compose([transforms])")
        - color: Specifies image-color format to convert to 
                 (default is L: 8-bit pixels, black and white)

        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset
        """
        key='data'
        self.color = color   
        self.inputData= readHDF5(inputData,key)
      
    def __len__(self):
        
        # Return the total number of data samples
        return self.inputData.shape[0]


    def __getitem__(self, ind):
        """Returns the image at the index 'ind' 
        (after applying transformations to the image, if specified).
        Params:
        -------
        - ind: (int) The index of the image to get
        Returns:
        --------
        - inputImage
        """
        image=self.inputData[ind,:,:]

        # Verify that image is in Tensor format
        if type(image) is not torch.Tensor:
            image=Image.fromarray(image)
            transform=transforms.ToTensor()
            image = transform(image)
        # Return the image and its label
        return image    

def create_only_inference_loader(batch_size,seed,test_data, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: for reproducibility
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Get create a Dataset object
#     testDataset = KneeDataset('blurredTest.h5','cleanTest.h5',transform)
    testDataset = KneeDatasetInference(test_data)

    # Dimensions and indices of training set
    dataset_size = len(testDataset)

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_test = SequentialSampler(list(range(dataset_size)))

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders

    test_loader = DataLoader(testDataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)


    # Return the training, validation, test DataLoader objects
    return test_loader