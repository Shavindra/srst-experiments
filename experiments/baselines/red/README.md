
# Data

## Data Source
1. Original images - KPN Maps dataset
2. Total images: 75 at 11100x11100
3. Multi class segmentation masks
```
class_colors = {
    'asphalt': (42,125,209),
    'clinkers': (41,220,188),
    'grass': (131,224,112),
    'moziak': (184,61,245),
    'bike-asphalt': (243,148,86),
    'cars': (250,250,55),
    'tiles': (255,96,55)
}
```
4. Semantic segmentation output:
    - Ground truth - RGB (3 Channels) PNG masks: 7 classes
    - Original images RGB (3 channels) JPG images: 75 images

# 2. Working Steps - Data Preprocessing
1. Get the images and masks
2. Preserve the original images and masks
3. Create masks for each class from the multi-class masks
4. Remove the images and masks if masks contain all black
5. Slice the images and masks into smaller images and masks
    - 512x512
    - 768x768
    - 1024x1024
    - 2048x2048
    - 3072x3072
6. Remove the images and masks if masks contain all black

# 3. Working Steps - EDA


# 3. Working Steps - Dataloader
2. Dataloader loads the the images (jpg images) and masks (png images)
3. Dataloader converts the images and masks to tensors
4. Visualise the images and masks by superimposing the masks on the images
5. Convert the RGB masks to single-channel masks where each pixel value is 0 (background) or 1 (asphalt)



## Steps to Use the ImageMaskDataset Class

1. **Import the necessary libraries**: This includes `os`, `torch`, `Dataset`, `DataLoader`, `transforms`, and `Image`.

2. **Define the ImageMaskDataset class**: This custom class should inherit from PyTorch's `Dataset` class.

3. **Initialize the ImageMaskDataset class**: In the `__init__` method, save the image and mask directories, any image transformations, and get a list of all the image files in the image directory.

4. **Override the `__len__` method**: This method should return the total number of images in the dataset.

5. **Override the `__getitem__` method**: This method should load and return a single image and its corresponding mask given an index. Construct the paths to the image and mask files, open and convert the images to RGB, apply any transformations if they exist, and then return the image and mask.

6. **Define a transforms.Compose object**: This object will be used to convert images to PyTorch tensors.

7. **Create an instance of ImageMaskDataset**: Use the image and mask directories and the transformation as parameters.

8. **Create a DataLoader**: This DataLoader will load images and masks in batches of 4 and shuffle them each epoch. The DataLoader can be iterated over to get batches of images and masks for training a model.



# 4. Working Steps - Models - UNet



# 5. Working Steps - Training

## 5.1. Training - UNet
 - Optimizer: Adam
    - Loss: BCEWithLogitsLoss
    - Learning Rate: 0.001
    - Batch Size: 4
    - Epochs: 100
- Activation: Sigmoid vs Softmax

- Metrics: 
    - Dice Coefficient
    - IoU
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    

## 5.2. Training - UNet++
1. Dataloader loads image as RGB - 3 channels 
2. Dataloader loads mask as single channel where each pixel value is 0 (background) or 1 (asphalt)
3. When training the input is an image of shape (B, C, H, W) 
    - B is the batch size, 
    - C is the number of channels, 
    - H is the height, 
    - W is the width. 
4. The output is a tensor of shape (B, 1, H, W) where each pixel value is 0 (background) or 1 (asphalt).
    - where each pixel value is between 0 and 1. 
    - During training the output is converted to a single channel mask where each pixel value is 0 (background) or 1 (asphalt).
5.During the validation step the mask is converted to a single channel mask where each pixel value is 0 (background) or 1 (asphalt). The output is a tensor of shape (B, 1, H, W) where each pixel value is 0 (background) or 1 (asphalt).


# 6. Working Steps - 