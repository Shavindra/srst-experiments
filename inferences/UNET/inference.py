# %%
import os
import sys
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/models')  # Adds the parent directory to the system path
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/utils')  # Adds the parent directory to the system path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# %%


import dataloader as dl

# %%
########################################## EXPERIMENT CONFIGURATION ##########################################

CLASS_NAME = 'asphalt'
EXPERIMENT_MODEL = 'UNET'
DATASET_VARIANT = 'binary_grayscale'
EXPERIMENT_VERSION_NUMBER = 1704589116

###############################################################################################################

IMG_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{CLASS_NAME}'
LABEL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{CLASS_NAME}'
VAL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{CLASS_NAME}'

EXPERIMENT_NAME= f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}'
EXPERIMENT_NAME_VERSION = f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}_{EXPERIMENT_VERSION_NUMBER}'

RESULT_DIR = f'runs/{EXPERIMENT_NAME_VERSION}'
LOG_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/logs'
TENSOIRBOARD_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/tensorboard'

MODEL_SAVE_PATH = f'runs/{EXPERIMENT_NAME_VERSION}/models'

MASK_COUNT = 99999


# %%
ANALYSIS_DIR = f'/home/sfonseka/dev/SRST/srst-analysis/{EXPERIMENT_NAME_VERSION}/analysis'
ANALYSIS_TEST_IMG_DIR = f'/home/sfonseka/dev/SRST/srst-analysis/test_images/{CLASS_NAME}'

########################################## BEST MODEL CONFIGURATION ##########################################

BEST_MODEL =  '/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/asphalt/preliminary/runs/UNET_binary_grayscale_asphalt_1704597437.863613/models/best_model_UNET_binary_grayscale_asphalt_1704597437.863613.pt'

state_dict = torch.load(
    BEST_MODEL,
    map_location=torch.device('cpu'))


from UNET import UNetBaseline
model = UNetBaseline(out_classes=1)

# Load the state dict into the model
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

###############################################################################################################

# %%
#Load the text file
with open(f'{ANALYSIS_DIR}/test_list.txt', 'r') as f:
    # Read the lines of the file
    test_files = f.readlines()

# %%
print(len(test_files))

# %%
from torch.utils.data import Dataset, DataLoader
import os, cv2
import numpy as np
from PIL import Image
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
# %%
import pickle


TEST_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/test/512/{CLASS_NAME}'
test_dataloader = dl.SRST_DataloaderGray(mask_dir=TEST_DIR, image_dir=IMG_DIR, mask_count=99999999999)
test_dataset = test_dataloader.dataset

# with open(f'{ANALYSIS_TEST_IMG_DIR}/test_list.txt', 'w') as f:
#     # Write the file paths to the file
#     for file_path in test_dataset.masks:
#         f.write(file_path + '\n')
#         print(file_path)


with open(f'{ANALYSIS_TEST_IMG_DIR}/{CLASS_NAME}_test_images.pickle', 'rb') as f:
    test_images_pickle = pickle.load(f)

print(f'Predictiing {len(test_images_pickle)}')

import os
import torch
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryAccuracy

# Initialize metrics
iou = IoU()
accuracy = BinaryAccuracy()

img_predictions = []
for img_to_pred in test_images_pickle:
    # print('predicting', img_to_pred['path'])
    print(filename)  # Outputs: 'file.txt'
    file_path = img_to_pred['path']
    filename = os.path.basename(file_path)

    img = img_to_pred['img']
    img_np = img_to_pred['img_np']

    # Assume we have some ground truth for each image
    ground_truth = ...

    # Make predictions
    with torch.no_grad():
        preds = model(img)

    # Compute and print IoU and accuracy
    iou_val = iou(preds, ground_truth)
    accuracy_val = accuracy(preds, ground_truth)
    print(f'IoU for {filename}: {iou_val}')
    print(f'Accuracy for {filename}: {accuracy_val}')


