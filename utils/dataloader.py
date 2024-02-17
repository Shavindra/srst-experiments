import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import utils as utils
import numpy as np
import random

random.seed(42)

from warnings import filterwarnings
filterwarnings("ignore")

class ImageMaskDatasetRGB(Dataset):
    """
    This Python code is used to create a custom PyTorch Dataset for loading and preprocessing 
    images and their corresponding masks from specified directories.

    The necessary libraries are imported. This includes os for interacting with the operating 
    system, torch for PyTorch operations, Dataset and DataLoader from torch.utils.data for 
    creating custom datasets and data loaders, transforms from torchvision for 
    image transformations, and Image from PIL for opening and manipulating images.
    """
    def __init__(self, image_dir, mask_dir, transform=None, mask_prefix='', mask_count=999999):

        """    
        In the __init__ method, the image and mask directories are saved, along with any 
        image transformations that are passed in. It also gets a list of all the image 
        files in the image directory.
        """

        # Assert paths
        assert os.path.isdir(image_dir), f'Image directory {image_dir} does not exist'
        assert os.path.isdir(mask_dir), f'Mask directory {mask_dir} does not exist'

        self.mask_prefix = mask_prefix
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = os.listdir(mask_dir)[:20]

        print('LEN', len(self.masks))

    def __len__(self):
        """
        The __len__ method returns the total number of images in the dataset.

        """
        return len(self.masks)

    def __getitem__(self, idx):
        """
        The __getitem__ method is used to load and return a single image and its corresponding
        mask given an index. It constructs the paths to the image and mask files, opens and 
        converts the images to RGB, applies any transformations if they exist, and then 
        returns the image and mask.
        """
        
        img_path = os.path.join(self.image_dir, self.masks[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx].replace('.jpg', '_mask.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # assuming mask is also RGB

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask



class ImageMaskDatasetGrayscale(ImageMaskDatasetRGB):

    def __getitem__(self, idx):
        # Load the image and mask

        img_path = os.path.join(self.image_dir, self.masks[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # Mask is already grayscale so no need to convert
        #mask = Image.open(mask_path).convert("L")  # assuming mask is also RGB

        # Convert mask to 1 channel grayscale and then apply binary threshold
        # mask = mask.convert("L")
        # mask = mask.point(lambda p: p > 0 and 1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, mask_path
  

class ImageSVMDataset(ImageMaskDatasetRGB):
    
    def __getitem__(self, idx):
        # Load the image and mask

        img_path = os.path.join(self.image_dir, self.masks[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # assuming mask is also RGB

            # Convert the image and mask to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        print('MASK SHAPE', mask_np.shape, 'IMAGE SHAPE', image_np.shape )
        print('MASK', mask_np, 'IMAGE', image_np)
        # Flatten the image array
        image_np = image_np.flatten()

        # Create a binary label from the mask (1 for asphalt, 0 for background)
        label = mask_np

        return image_np, label

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create a class for dataloader
class SRST_Dataloader():

    def __init__(self, image_dir=None, mask_dir=None, transform=transform, mask_count=999999):

        self.dataset = ImageMaskDatasetRGB(image_dir=image_dir, mask_dir=mask_dir, transform=transform, mask_count=mask_count)
        self.data_loader = DataLoader(self.dataset, batch_size=8, num_workers=4)
        pass 

    
class SRST_DataloaderGray():

    def __init__(self, image_dir=None, mask_dir=None, transform=transform, mask_count=999999):
        self.dataset = ImageMaskDatasetGrayscale(image_dir=image_dir, mask_dir=mask_dir, transform=transform, mask_count=mask_count)
        # Batch size changed to 12 and num_workers 8
        self.data_loader = DataLoader(self.dataset, batch_size=12, num_workers=8)
        pass

class SRST_DataloaderSVM():
    
        def __init__(self, image_dir='path/to/images', mask_dir='path/to/masks', transform=transform):
            self.dataset = ImageSVMDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
            self.data_loader = DataLoader(self.dataset, batch_size=12, num_workers=8)
            pass