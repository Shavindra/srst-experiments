import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import utils as utils
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
    def __init__(self, image_dir, mask_dir, transform=None, mask_prefix=''):

        """    
        In the __init__ method, the image and mask directories are saved, along with any 
        image transformations that are passed in. It also gets a list of all the image 
        files in the image directory.
        """
        self.mask_prefix = mask_prefix
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = os.listdir(mask_dir)[:200]

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

        return image, mask
  
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create a class for dataloader
class SRST_Dataloader():

    def __init__(self, image_dir='path/to/images', mask_dir='path/to/masks', transform=transform):
        self.dataset = ImageMaskDatasetRGB(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
        self.data_loader = DataLoader(self.dataset, batch_size=4, num_workers=2)
        pass 

    
class SRST_DataloaderGray():

    def __init__(self, image_dir='path/to/images', mask_dir='path/to/masks', transform=transform):
        self.dataset = ImageMaskDatasetGrayscale(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
        self.data_loader = DataLoader(self.dataset, batch_size=4, num_workers=2)
        pass