import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import utils as utils
import numpy as np


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
    def __init__(self, image_dir, mask_dir, transform=None, mask_prefix='', mask_size=100):

        """    
        In the __init__ method, the image and mask directories are saved, along with any 
        image transformations that are passed in. It also gets a list of all the image 
        files in the image directory.
        """
        self.mask_prefix = mask_prefix
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = os.listdir(mask_dir)[:mask_size]
        print(f'Found {len(self.masks)} masks in {mask_dir}')
        

        print('LEN', len(self.masks))

    def __len__(self):
        """
        The __len__ method returns the total number of images in the dataset.

        """
        return len(self.masks)


class ImageSVMDataset(ImageMaskDatasetRGB):
    
    def __getitem__(self, idx):
        # Load the image and mask

        img_path = os.path.join(self.image_dir, self.masks[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
            # Convert the image and mask to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Flatten the image array
        image_np = image_np.reshape(-1, 3)  # The -1 will be replaced with the total number of pixels in the image

        # Create a binary label from the mask (1 for asphalt, 0 for background)
        label = mask_np.flatten()

        # print('MASK SHAPE', mask_np.shape)
        # print('IMAGE SHAPE', image_np.shape)
        # print('LABEL SHAPE', label.shape)

        return image_np, label

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

class SRST_DataloaderSVM():
    
        def __init__(self, image_dir='path/to/images', mask_dir='path/to/masks', transform=transform, mask_size=100):
            self.dataset = ImageSVMDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, mask_size=mask_size)
            self.data_loader = DataLoader(self.dataset, batch_size=1, num_workers=2)
            pass