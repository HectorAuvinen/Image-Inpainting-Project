
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import glob
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import pickle


#Set the values for min and max of offset and spacing
MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6

#This function is used to resize the images to 100 x 100
def resize(thing,im_shape):

    resize_transforms = transforms.Compose([
    transforms.Resize(size=im_shape),
    transforms.CenterCrop(size=(im_shape, im_shape)),
    ])
    image = np.array(resize_transforms(Image.open(thing)))
    return image

#This function takes an image, offset and spacing and returns image array, known array and target array
def ex4(image_array: np.ndarray, offset: tuple, spacing: tuple):
    target_image = image_array.copy()  
    # Change dimensions from (H, W, C) to PyTorch's (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1
    
    known_pixels = np.sum(known_array[0], dtype=np.uint32)
    
    # Create target_array - don't forget to use .copy(), otherwise target_array
    # and image_array might point to the same array!
    target_array = image_array[known_array == 0].copy()
    
    # Use image_array as input_array
    image_array[known_array == 0] = 0
    
    return (target_image,image_array, known_array, target_array)

def grid(image_array: np.ndarray, offset: tuple, spacing: tuple):
    target = image_array.copy()  # copy to avoid modifying the original image
    known_array = np.zeros_like(image_array)  # array to keep track of known pixels
    known_array[
        :, offset[1] :: spacing[1], offset[0] :: spacing[0]
    ] = 1  # set known pixels to 1
    target_array = image_array[known_array == 0].copy()  # array of unknown pixels
    image_array[known_array == 0] = 0  # set unknown pixels to 0
    return (
        target,
        image_array,
        known_array,
        target_array,
    )  # return target, image, known, target_array


#This class is used to get the data paths & indices (created so that the data can be split into train,val,test sets)
class PathDataset(Dataset):
    def __init__(self,image_dir):
        #Get all the image paths
        self.data = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #get image path at index
        image_path = self.data[index]

        #return image path,index
        return image_path,index

#This class is used to read an image, apply crop, apply ex4, and return the (stacked) input array, full image and index for dataloader
class ImageDataset(Dataset):
    def __init__(self, dataset: Dataset):
        #define dataset
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        #get the image with index
        image_data, index = self.dataset[index]
        #open and crop image
        image_data = resize(image_data,100)
        #store normalized full image
        full_image = image_data / 255

        #store normalized image
        image = image_data / 255

        #define offset and spacing so that we can get input,known and target array with ex4
        offset = (np.random.randint(MIN_OFFSET,MAX_OFFSET),np.random.randint(MIN_OFFSET,MAX_OFFSET))
        spacing = (np.random.randint(MIN_SPACING,MAX_SPACING),np.random.randint(MIN_SPACING,MAX_SPACING))

        ######################
        target_image, input_array, known_array, target_array = ex4(image, offset, spacing)
        #print(TF.to_tensor(input_array).shape)
        #print(TF.to_tensor(known_array[0:1,:,:]).shape)
        # stack the input array and known array (to feed more essential information to the model)
        return_array = torch.cat((TF.to_tensor(input_array), TF.to_tensor(known_array[0:1,:,:])), dim=1)
        return_array = torch.transpose(return_array,0,1).type(torch.FloatTensor)

        # undo transpose (on full image) from ex4 & convert to torch tensor
        full_image = torch.from_numpy(full_image.transpose(2, 1, 0)).type(torch.FloatTensor)
        ######################
        #return_array = return_array.type(torch.FloatTensor)
        #full_image = full_image.type(torch.FloatTensor)
        return return_array, full_image, index