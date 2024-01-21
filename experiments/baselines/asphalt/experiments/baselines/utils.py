import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from PIL import Image

random.seed(42)

def visualize_random_image_mask(dataset, idx=None):
    """
    Function to retrieve and visualize a random image and its corresponding mask from a dataset.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to retrieve the image and mask from.
    """
    # Get a random index
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    print(f'Displaying the image idx {idx}')
    # Retrieve the image and mask
    image, mask = dataset[idx]

    # Convert the image and mask tensors to numpy arrays for visualization
    image = image.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0).numpy()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Image')

    # Display the mask in the second subplot
    axs[1].imshow(mask)
    axs[1].set_title('Mask')

    # Show the figure
    plt.show()


def visualize_random_image_mask_cv2(dataset, transparency=0.5):
    """
    Function to retrieve and visualize a random image and its corresponding mask from a dataset using cv2.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to retrieve the image and mask from.
    transparency (float): The transparency of the mask when superimposed on the image. Should be between 0 (completely transparent) and 1 (completely opaque).
    """
    # Get a random index
    idx = random.randint(0, len(dataset) - 1)

    # Retrieve the image and mask
    image, mask = dataset[idx]

    # Convert the image and mask tensors to numpy arrays for visualization
    # Also, convert the images from RGB to BGR as cv2 uses BGR format
    image = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)

    # Superimpose the image and mask
    superimposed = cv2.addWeighted(image, 1-transparency, mask, transparency, 0)

    # Display the superimposed image
    cv2.imshow('Superimposed Image', superimposed)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def superimpose_images(image, mask, transparency=0.5, grayscale=False):
    """
    Function to superimpose an image and a mask and visualize the result.

    Parameters:
    image (tensor): The image tensor.
    mask (tensor): The mask tensor.
    transparency (float): The transparency of the mask when superimposed on the image. Should be between 0 (completely transparent) and 1 (completely opaque).
    """

    # Convert the image and mask tensors to numpy arrays for visualization
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.permute(1, 2, 0).numpy()

    # Normalize the image and mask to the range [0, 1]
    image_np = image_np / image_np.max()
    mask_np = mask_np / mask_np.max()

    # Convert the normalized image and mask to the range [0, 255]
    image_np = (image_np * 255).astype(np.uint8)
    mask_np = (mask_np * 255).astype(np.uint8)

    if grayscale == True:
        # Convert the grayscale mask to RGB
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)

    # Superimpose the image and mask using cv2.addWeighted
    superimposed = cv2.addWeighted(image_np, 1, mask_np, transparency, 0)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image, mask, and superimposed image
    axs[0].imshow(image_np)
    axs[0].set_title('Original Image')

    axs[1].imshow(mask_np)
    axs[1].set_title('Mask')

    axs[2].imshow(superimposed)
    axs[2].set_title('Superimposed Image')

    # Remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def visualise_svm(image, label, original_shape=(512, 512, 3)):
    """
    Function to visualize an image and its corresponding label.

    Parameters:
    image (tensor): The image tensor.
    label (2D array): The label.
    original_shape (tuple): The original shape of the image.
    """
    # Convert the image tensor to a numpy array for visualization
    if isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = image.permute(1, 2, 0).numpy()

    # Reshape the image back to its original shape
    image_np = image_np.reshape(original_shape)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    axs[0].imshow(image_np)
    axs[0].set_title('Image')

    # Display the label
    axs[1].imshow(label, cmap='gray')
    axs[1].set_title('Label')

    # Show the figure
    plt.show()

class Visualiser():
    def __init__(self, dataloader):
        self.dataset = dataloader.dataset
        pass

    def vis_random(self, idx=None):
        visualize_random_image_mask(self.dataset, idx=idx)

    
    def vis_cv2(self, transparency=0.5):
        visualize_random_image_mask_cv2(self.dataset, transparency=transparency)

    def superimpose_images(self, idx=None, alpha=0.5, grayscale=False):
        if idx is None:
            idx = random.randint(0, len(self.dataset) - 1)

        print(idx)
        image, mask = self.dataset[idx]

        superimpose_images(image, mask, transparency=alpha, grayscale=grayscale)


class VisualiserSVM():
    def __init__(self, dataloader):
        self.dataset = dataloader.dataset
        pass

    def vis_random(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self.dataset) - 1)

        print(idx)
        image, mask = self.dataset[idx]



        visualise_svm(image, mask)