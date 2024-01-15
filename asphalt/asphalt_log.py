# %%
import torch
import torch.nn as nn
import torch.optim as optim
import dataloader  as dl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from torchmetrics.classification import BinaryJaccardIndex as IoU

from models.UNET import UNetBaseline

# %%

class_name = 'asphalt'
image_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{class_name}'
label_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{class_name}'
val_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{class_name}'

# %%
dataloader = dl.SRST_DataloaderGray(mask_dir=label_dir, image_dir=image_dir)
val_dataloader = dl.SRST_DataloaderGray(mask_dir=val_dir, image_dir=image_dir)


# %%
from tqdm import tqdm

import time

timestamp = int(time.time())
# [Your import statements and other code]
writer = SummaryWriter(f'runs_log/unet_grayscale_experiment/{class_name}/tensorboard/UNet_{class_name}/exp_{timestamp}')


best_loss = float('inf')

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    print('Training')
    model.train()
    total_loss = 0
    iou_metric = IoU()  # Initialize IoU metric for binary classification

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update IoU metric
        preds = (outputs > THRESHOLD).int()  # Convert outputs to binary predictions
        iou_metric.update(preds, masks)

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    iou_score = iou_metric.compute()  # Compute IoU score for the epoch

    return avg_loss, iou_score

THRESHOLD = 0.35  # Adjust as needed

def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    iou_metric = IoU()  # Initialize IoU for binary classification (background, asphalt)

    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Update IoU metric
            # For binary classification, you can use a threshold to convert outputs to binary format
            preds = (outputs > THRESHOLD).int()  # Adjust THRESHOLD as needed, e.g., 0.5
            iou_metric.update(preds, masks)

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader)
    iou_score = iou_metric.compute()  # Compute final IoU score

    return avg_loss, iou_score


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = f'runs_logs/unet_grayscale_experiment/{class_name}/models'
os.makedirs(model_save_path, exist_ok=True)

model = UNetBaseline(out_classes=1).to(device)  # For grayscale, out_classes should be 1
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_loader = dataloader.data_loader
val_loader = val_dataloader.data_loader

num_epochs = 20
for epoch in tqdm(range(num_epochs), desc='Epochs'):  # tqdm wrapper for epochs
    train_loss, train_metric = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_metric = eval_model(model, val_loader, criterion, device)


    # Save the model if it's the best one so far
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(model_save_path, f'best_model_unet_{class_name}.pt'))
        print(f'Saved new best model with loss {best_loss:.4f}')

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
    writer.add_scalars('IoU', {'Train': train_metric, 'Validation': val_metric}, epoch)

writer.close()


