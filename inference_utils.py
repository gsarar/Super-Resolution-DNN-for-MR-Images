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
from math import log10
import torch.nn as nn

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