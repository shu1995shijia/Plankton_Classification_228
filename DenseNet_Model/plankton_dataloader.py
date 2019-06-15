# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data

# Uncomment for Python2
# from __future__ import print_function



class PlanktonDataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.

    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """
    
    def __init__(self, root_dir ,file_dir ,transform=transforms.ToTensor(), color='L'):
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
        - image_dir: The absolute filepath to the dataset on ieng6
        - image_info: A Pandas DataFrame of the dataset metadata
        - image_filenames: An array of indices corresponding to the images
        - labels: An array of labels corresponding to the each sample
        - classes: A dictionary mapping each disease name to an int between [0, 13]
        """
        
        self.transform = transform
        self.color = color
        self.file_dir = root_dir+file_dir
        self.label_dir = "labels.txt"
        
        f = open(self.label_dir,"r")
        # And for reading use
        classes = f.read()
        f.close()
        classes_list = classes.split('\n')
        file_dir_list = file_dir.split('.')
        index_class = classes_list.index(file_dir_list[0])
        self.length = int(classes_list[index_class-1])
        self.label = self.oneHot(int(index_class//2))
        
        
    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """
        self.images = np.load(self.file_dir)
        image =  self.images[ind]
        image = image.reshape((128,128,1))
        #If a transform is specified, apply it
        if self.transform is not None:
            image = self.transform(image)
            
        # Verify that image is in Tensor format
        #if type(image) is not torch.Tensor:
        #    image = transform.ToTensor(image)

        # Convert multi-class label into binary encoding 

        label = self.label
        
        # Return the image and its label
        return (image, label)
    
    def oneHot(self, number, max_val=345):
        """ Computes onehot.
        Input: Y_oh: list of number
              max_val: The max one-hot size
        Returns: 2D list correspind to the each label's one hot representation
        """
        onehot = [0] * (int(max_val))
        onehot[number] = 1
        return np.array(onehot)
    

def create_split_loaders(root_dir, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True, 
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
    
    
    
    list_of_datasets = []
    for j in os.listdir(root_dir):
        if not j.endswith('.npy'):
            continue  # skip non-json files
        list_of_datasets.append(PlanktonDataset(root_dir=root_dir, file_dir=j, transform=transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])))
    # once all single json datasets are created you can concat them into a single one:
    quickdraw_dataset = data.ConcatDataset(list_of_datasets)
    
    

    # Dimensions and indices of training set
    dataset_size = len(quickdraw_dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(quickdraw_dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(quickdraw_dataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)

    val_loader = DataLoader(quickdraw_dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)

# Reference based on CSE 253 Assignment 3