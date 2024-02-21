# %%
# %%
import os
import sys
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/models')  # Adds the parent directory to the system path
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/utils')  # Adds the parent directory to the system path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os, cv2
import numpy as np
from PIL import Image
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryAccuracy

import dataloader as dl

# %%
import torch
import matplotlib.pyplot as plt
import torch.nn as nn  # Make sure this import statement is included


# %%
THRESHOLD = 0.5

experiment_versions = [
    'UNET_binary_grayscale_asphalt_MASKED_METRICS_1708018219.243841',
    'UNET_binary_grayscale_clinkers_MASKED_METRICS_1708020065.677586',
    'UNET_binary_grayscale_bike-asphalt_MASKED_METRICS_1708018220.449231',
    'UNET_binary_grayscale_grass_MASKED_METRICS_1708016508.855831',
    'UNET_binary_grayscale_mozaik_MASKED_METRICS_1708016451.646911',
    'UNET_binary_grayscale_tiles_MASKED_METRICS_1708015922.418231'
]

MODEL_VERSIONS = []

for version in experiment_versions:
    MODEL_VERSIONS.append({
        'CLASS_NAME': version.split('_')[3],
        'MODEL': f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/asphalt/runs/{version}/models/best_iou_model_{version}.pt',
        'IOU_MODEL': f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/asphalt/runs/{version}/models/best_mask_iou_model_{version}.pt'
    })


import csv

test_results = []

activations = {}

def register_hooks(module):
    if isinstance(module, nn.Conv2d):
        def hook(module, input, output):
            activations[module] = output.detach()
        module.register_forward_hook(hook)


# %%


def test_model(model, test_loader, device, class_name, model_path):
    model.eval()

    # Open CSV file for writing
    with open('test_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_filename', 'test_iou', 'test_accuracy', 'class_name', 'model_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        progress_bar = tqdm(test_loader, desc='Testing', leave=False)
        with torch.no_grad():
            for batch_index, (test_images, test_masks, __paths) in enumerate(progress_bar):
                test_images, test_masks = test_images.to(device), test_masks.to(device)

                for i in range(test_images.size(0)):
                    image = test_images[i].unsqueeze(0)
                    mask = test_masks[i].unsqueeze(0)
                    # print file name
                    image_filename = os.path.basename(__paths[i])
                    print(image_filename)

                    # Plot the image and mask
                    img_np = image.squeeze(0).cpu().numpy().transpose((1, 2, 0))
                    
                    output = model(image)
                    output_sigmoid = torch.sigmoid(output)

                    # Convert the output to a binary mask
                    test_pred = (output_sigmoid > THRESHOLD).int()  # Adjust THRESHOLD as needed, e.g., 0.5

                    metric_test_iou = IoU().to(device)  # Initialize IoU for binary classification (background, asphalt)
                    metric_test_accuracy = BinaryAccuracy().to(device)  # Initialize accuracy metric for binary classification

                    metric_test_iou.update(test_pred, mask)
                    metric_test_accuracy.update(test_pred, mask)

                    score_test_iou = metric_test_iou.compute()  # Compute final IoU score
                    score_test_accuracy = metric_test_accuracy.compute()  # Compute final accuracy score

                    # Write metrics to CSV file
                    writer.writerow({'image_filename': image_filename, 'test_iou': score_test_iou.item(), 'test_accuracy': score_test_accuracy.item(), 'class_name': class_name, 'model_path': model_path})

                    # Append results to list
                    test_results.append({
                        'image_filename': image_filename, 
                        'test_iou': score_test_iou.item(), 
                        'test_accuracy': score_test_accuracy.item(), 
                        'class_name': class_name, 
                        'model_path': model_path,
                        'output': output,
                        'pred': test_pred,
                        'mask': mask,
                        'image_np': img_np,   
                    })
           


# %%
from UNET import UNetBaseline
model = UNetBaseline(out_classes=1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', DEVICE)

for item in MODELS_LIST:
    CLASS_NAME = 'mozaik'
    BEST_MODEL = item['BEST_MODEL']

    IMG_DIR = '/projects/0/gusr51794/srst_scratch_drive/binary_training/temp/images'
    TEST_DIR = '/projects/0/gusr51794/srst_scratch_drive/binary_training/temp/masks'

    model.apply(register_hooks)

    state_dict = torch.load(BEST_MODEL, map_location=torch.device('cpu'))['model_state_dict']


    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    test_dataloader = dl.SRST_DataloaderGray(mask_dir=TEST_DIR, image_dir=IMG_DIR, mask_count=1)
    test_dataset = test_dataloader.dataset

    test_model(model, test_dataloader.data_loader, DEVICE, class_name=CLASS_NAME, model_path=BEST_MODEL)


# Save the test results to a pickle file
with open('test_results.pkl', 'wb') as f:
    pickle.dump(test_results, f)

# %%
# Load the results from pickle
import pickle
with open('test_results.pkl', 'rb') as f:
     test_results = pickle.load(f)


# %%
# Get all the keys from test_results
keys = test_results
keys

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    # print(pred)
    print('---' * 10)

    pred = item['pred'].int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = item['pred'].squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = item['pred'].int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = item['pred'].squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = (output_sigmoid > 0.5).int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = output_sigmoid.squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = (output_sigmoid > 0.5).int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = output_sigmoid.squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = (output_sigmoid > threshold).int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = output_sigmoid.squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = (output_sigmoid > threshold).int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = output_sigmoid.squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
for item in test_results:
    print(item['image_filename'], item['test_iou'], item['test_accuracy'])

    output_sigmoid = torch.sigmoid(item['output'])

    print('output', output_sigmoid)
    print('output min', output_sigmoid.min())
    print('output max', output_sigmoid.max())

    threshold = output_sigmoid.median() 
    
    # Plot the image, mask, and prediction
    plt.figure(figsize=(20, 10))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(item['image_np'])
    plt.title('Image')
    plt.axis('off')

    # Plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(item['mask'].squeeze())  # Assuming the mask is stored in 'mask_np'
    plt.title('Mask')
    plt.axis('off')

    # Plot the prediction
    plt.subplot(1, 3, 3)

    print(pred)
    print('---' * 10)

    pred = (output_sigmoid > threshold).int().squeeze(0).squeeze(0).cpu().numpy()

    print(pred)

    plt.imshow(pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()


        # Convert the prediction to a numpy array and remove extra dimensions
    prediction_np = output_sigmoid.squeeze().cpu().numpy()

    # Display the distribution of the values in the prediction
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_np.flatten(), bins=50, color='c')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
# Function to visualize the activations
def visualize_activation(activation, num_cols=8):
    num_kernels = activation.size(1)
    num_rows = num_kernels // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            ax.imshow(activation[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
    plt.show()

# Visualize activations for each layer
for layer, activation in activations.items():
    print(f"Visualizing activations for layer: {layer}")
    visualize_activation(activation)


