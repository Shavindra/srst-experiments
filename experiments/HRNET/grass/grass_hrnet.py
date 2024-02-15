# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append('../../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader

# Now you can import your module
from models.hrnet import HighResolutionNetBaseline
from utils import dataloader as dl

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryROC


# LOGGING
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
import csv
from filelock import FileLock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', DEVICE)
# print("CUDA available: ", torch.cuda.is_available())
# print("Current device: ", torch.cuda.current_device())
# print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))



# Ignore warnings
from warnings import filterwarnings
filterwarnings("ignore")

# Get the current date and time
from datetime import datetime, timedelta
now = datetime.now()

# 14 days ago
now_before = datetime.now() - timedelta(days=14)
now_before = now_before.timestamp()
now = now_before
# %%

CLASS_NAME = 'grass'
EXPERIMENT_MODEL = 'HRNET'
DATASET_VARIANT = 'binary_grayscale'

IMG_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{CLASS_NAME}'
LABEL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{CLASS_NAME}'
VAL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{CLASS_NAME}'

EXPERIMENT_NAME= f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}'
EXPERIMENT_NAME_VERSION = f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}_{now}'

RESULT_DIR = f'runs/{EXPERIMENT_NAME_VERSION}'
LOG_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/logs'
TENSOIRBOARD_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/tensorboard'

MODEL_SAVE_PATH = f'runs/{EXPERIMENT_NAME_VERSION}/models'

# Create directories to save results
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)



# Load yaml file
import yaml

with open('/home/sfonseka/dev/SRST/srst-dataloader/models/configs/hrnet_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



# %%
from tqdm import tqdm

# [Your import statements and other code]

#writer = TensorBoardLogger(TENSOIRBOARD_DIR, name=EXPERIMENT_NAME)

writer = SummaryWriter(TENSOIRBOARD_DIR)

# Create a CSV file and write the headers
MODEL_RESULT_FILE = f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/{EXPERIMENT_MODEL}/experiment_results.csv'
lock = FileLock(f"/home/sfonseka/dev/SRST/srst-dataloader/experiments/{EXPERIMENT_MODEL}/experiment_results.lock")
with open(f'{MODEL_RESULT_FILE}', 'a', newline='') as file:
    exp_stats = csv.writer(file)
#    exp_stats.writerow(["Experiment", "Version", "Epoch", "Train Loss", "Validation Loss", "Train IoU", "Validation IoU", "Train Accuracy", "Validation Accuracy"])

best_loss = float('inf')
best_iou = 0

EPOCHS = 3
THRESHOLD = 0.5  # Adjust as needed
MASK_COUNT = 20

LR = 0.001

dataloader = dl.SRST_DataloaderGray(mask_dir=LABEL_DIR, image_dir=IMG_DIR, mask_count=MASK_COUNT)
val_dataloader = dl.SRST_DataloaderGray(mask_dir=VAL_DIR, image_dir=IMG_DIR, mask_count=MASK_COUNT)

# Example of model instantiation
model = HighResolutionNetBaseline(config=config)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_loader = dataloader.data_loader
val_loader = val_dataloader.data_loader

# %%
print('CLASS_NAME: ', CLASS_NAME)
print('EXPERIMENT_MODEL: ', EXPERIMENT_MODEL)
print('EXPERIMENT_NAME: ', EXPERIMENT_NAME)
print('EXPERIMENT_NAME_VERSION: ', EXPERIMENT_NAME_VERSION)
print('RESULT_DIR: ', RESULT_DIR)
print('LOG_DIR: ', LOG_DIR)
print('TENSOIRBOARD_DIR: ', TENSOIRBOARD_DIR)
print('MODEL_SAVE_PATH: ', MODEL_SAVE_PATH)
print('MODEL_RESULT_FILE: ', MODEL_RESULT_FILE)
print('EPOCHS: ', EPOCHS)
print('THRESHOLD: ', THRESHOLD)
print('MASK_COUNT: ', MASK_COUNT)
print('LR: ', LR)
print('DEVICE: ', DEVICE)
print('optimizer: ', optimizer.__class__.__name__)
print('criterion: ', criterion.__class__.__name__)
print('CONFIG: ', config)

epoch_number = 0
best_loss = float('inf')
best_iou = 0

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    print('Training')
    model.train()
    total_loss = 0
    
    metric_iou = IoU().to(device)  # Initialize IoU metric for binary classification
    metric_accuracy = BinaryAccuracy().to(device)  # Initialize accuracy metric for binary classification
    metric_auroc = BinaryAUROC().to(device)  # Initialize AUROC metric for binary classification
    metric_precision = BinaryPrecision().to(device)  # Initialize precision metric for binary classification
    metric_recall = BinaryRecall().to(device)  # Initialize recall metric for binary classification
    metric_f1 = BinaryF1Score().to(device)  # Initialize F1 metric for binary classification
    metric_roc = BinaryROC().to(device)  # Initialize ROC metric for binary classification

    new_metrics = [
        {'name': 'iou', 'metric': metric_iou},
        {'name': 'accuracy', 'metric': metric_accuracy},
        {'name': 'auroc', 'metric': metric_auroc},
        {'name': 'precision', 'metric': metric_precision},
        {'name': 'recall', 'metric': metric_recall},
        {'name': 'f1', 'metric': metric_f1},
        {'name': 'roc', 'metric': metric_roc},
    ]

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, masks, ___path in progress_bar:
        images, masks = images.to(device), masks.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = model(images)


        # Compute the loss and its gradients
        loss = criterion(outputs, masks)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        total_loss += loss.item()

        # Update IoU metric
        outputs_sigmoid = torch.sigmoid(outputs)  # Apply sigmoid to convert to probabilities
        preds = (outputs_sigmoid > THRESHOLD).int()  # Convert outputs to binary predictions

        metric_iou.update(preds, masks)
        metric_accuracy.update(preds, masks)

        for metric in new_metrics:
            metric['metric'].update(preds, masks)

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    
    new_metric_scores = [{'metric_name': 'train_loss', 'metric_score': avg_loss}]
    
    for metric in new_metrics:
        if metric['name'] == 'iou' or metric['name'] == 'accuracy':
            new_metric_scores.append({'metric_name': f'train_{metric["name"]}', 'metric_score': metric['metric'].compute()})
        else:
            new_metric_scores.append({'metric_name': f'train_{metric["name"]}', 'metric_score': metric['metric'].compute(outputs_sigmoid, masks)})


    print('EPOCH METRICS', new_metric_scores)
    return new_metric_scores


def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0


    metric_eval_iou = IoU().to(device)  # Initialize IoU metric for binary classification
    metric_eval_accuracy = BinaryAccuracy().to(device)  # Initialize accuracy metric for binary classification
    metric_eval_auroc = BinaryAUROC().to(device)  # Initialize AUROC metric for binary classification
    metric_eval_precision = BinaryPrecision().to(device)  # Initialize precision metric for binary classification
    metric_eval_recall = BinaryRecall().to(device)  # Initialize recall metric for binary classification
    metric_eval_f1 = BinaryF1Score().to(device)  # Initialize F1 metric for binary classification
    metric_eval_roc = BinaryROC().to(device)  # Initialize ROC metric for binary classification

    new_eval_metrics = [
        {'name': 'iou', 'metric': metric_eval_iou},
        {'name': 'accuracy', 'metric': metric_eval_accuracy},
        {'name': 'auroc', 'metric': metric_eval_auroc},
        {'name': 'precision', 'metric': metric_eval_precision},
        {'name': 'recall', 'metric': metric_eval_recall},
        {'name': 'f1', 'metric': metric_eval_f1},
        {'name': 'roc', 'metric': metric_eval_roc},
    ]

    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for img, msk, _p in progress_bar:
            eval_images, eval_masks = img.to(device), msk.to(device)

            outputs = model(eval_images)

            loss = criterion(outputs, eval_masks)
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            outputs_sigmoid = torch.sigmoid(outputs)

            # Update IoU metric
            # For binary classification, you can use a threshold to convert outputs to binary format
            eval_preds = (outputs_sigmoid > THRESHOLD).int()  # Adjust THRESHOLD as needed, e.g., 0.5


            metric_eval_iou.update(eval_preds, eval_masks)
            metric_eval_accuracy.update(eval_preds, eval_masks)

            for metric in new_eval_metrics:
                metric['metric'].update(outputs_sigmoid, eval_masks.float())

            print('metric_eval_iou: ', metric_eval_iou.compute())

            progress_bar.set_postfix(loss=loss.item())

    avg_eval_loss = total_loss / len(val_loader)
    
    new_eval_metric_scores = [{'metric_name': 'train_loss', 'metric_score': avg_eval_loss}]
    
    for metric in new_eval_metrics:
        if metric['name'] == 'iou' or metric['name'] == 'accuracy':
            new_eval_metric_scores.append({'metric_name': f'train_{metric["name"]}', 'metric_score': metric['metric'].compute()})
        else:
            new_eval_metric_scores.append({'metric_name': f'train_{metric["name"]}', 'metric_score': metric['metric'].compute(outputs_sigmoid, eval_masks)})

    # metrics = {
    #     'eval_loss': avg_eval_loss,
    #     'eval_iou': score_eval_iou,
    #     'eval_accuracy': score_eval_accuracy,
    # }

    print('EPOCH METRICS', new_eval_metric_scores)

    return new_eval_metric_scores


import time

# Record the start time
start_time = time.time()
patience = 15  # Number of epochs to wait for improvement before stopping
best_score = 0  # Best score achieved so far
wait = 0  # Number of epochs we have waited so far without improvement

for epoch in tqdm(range(EPOCHS), desc='Epochs'):  # tqdm wrapper for epochs
    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device=DEVICE, threshold=THRESHOLD)
    val_metrics = eval_model(model, val_loader, criterion, device=DEVICE, threshold=THRESHOLD)

    train_loss = train_metrics['train_loss']
    train_metric_iou = train_metrics['train_iou'].item()
    val_loss = val_metrics['eval_loss']
    val_metric_iou = val_metrics['eval_iou'].item()

    train_metric_accuracy = train_metrics['train_accuracy'].item()
    val_metric_accuracy = val_metrics['eval_accuracy'].item()

    logging_step = epoch_number + 1

    # print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    # print(f'Epoch {epoch}, Logging Step: {logging_step}, Train IoU: {train_metric_iou}, Val IoU: {val_metric_iou}')
    # print(f'Epoch {epoch}, Logging Step: {logging_step}, Train Accuracy: {train_metric_accuracy}, Val Accuracy: {val_metric_accuracy}')
    
    # writer.add_scalars('Training Loss vs Validation Loss', {'train': train_loss, 'val': val_loss}, logging_step, walltime=now_before)
    # writer.add_scalars('Training IoU vs Validation IoU', {'train': train_metric_iou, 'val': val_metric_iou}, logging_step, walltime=now_before)
    # writer.add_scalars('Training Accuracy vs Validation Accuracy', {'train': train_metric_accuracy, 'val': val_metric_accuracy}, logging_step, walltime=now_before)


    for metric in train_metrics:
        print(f'Epoch {epoch}, Logging Step: {logging_step}, Train {metric}: {train_metrics[metric]}, Val {metric}: {val_metrics[metric]}')
        writer.add_scalar(f'Training/{metric}', train_metrics[metric], epoch, walltime=now_before)
        writer.add_scalar(f'Validation/{metric}', val_metrics[metric], epoch, walltime=now_before)
        writer.add_scalar(f'Training {metric} vs Validation {metric}', {'train': train_metrics[metric], 'val': val_metrics[metric]}, epoch, walltime=now_before)

    # writer.add_scalar('Metrics/Train_Loss', train_loss, logging_step, walltime=now_before)
    # writer.add_scalar('Metrics/Val_Loss', val_loss, logging_step, walltime=now_before)

    # writer.add_scalar('Metrics/Train_IoU', train_metric_iou, logging_step, walltime=now_before)
    # writer.add_scalar('Metrics/Val_IoU', val_metric_iou, logging_step, walltime=now_before)

    # writer.add_scalar('Metrics/Train_Accuracy', train_metric_accuracy, logging_step, walltime=now_before)
    # writer.add_scalar('Metrics/Val_Accuracy', val_metric_accuracy, logging_step, walltime=now_before)

    epoch_number += 1

    # Write the metrics for this epoch to the CSV file
    with lock:
        with open(f'{MODEL_RESULT_FILE}', 'a', newline='') as file:
            exp_stats = csv.writer(file)
            exp_stats.writerow([
                EXPERIMENT_NAME,
                EXPERIMENT_NAME_VERSION,
                epoch_number,
                train_loss,
                val_loss,
                train_metric_iou,
                val_metric_iou,
                train_metric_accuracy,
                val_metric_accuracy,

            ])

            print('Wrote metrics to CSV file: ', f'{MODEL_RESULT_FILE}')


    writer.flush()
    # Save the model if it's the best one so far
    # Save the model if it's the best one so far in terms of IoU
    if val_metric_iou > best_iou:
        best_iou = val_metric_iou
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'best_model_{EXPERIMENT_NAME_VERSION}.pt'))
        print(f'Saved new best model with IoU {best_iou:.4f}')
    else:
        wait +=1

    # If we have waited for `patience` epochs without improvement, stop training
    if wait >= patience:
        print("Early stopping")
        break

writer.close()


# Compute the total running time
total_time = time.time() - start_time

# Create a CSV file and write the headers and values
with open(f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/{EXPERIMENT_MODEL}/experiment_setup.csv', 'a', newline='') as file:
    stats = csv.writer(file)
    stats.writerow([
        EXPERIMENT_NAME,
        EXPERIMENT_NAME_VERSION,
        IMG_DIR,
        LABEL_DIR,
        VAL_DIR,
        CLASS_NAME,
        EPOCHS,
        THRESHOLD,
        MASK_COUNT,
        DEVICE,
        LR,
        optimizer.__class__.__name__,
        criterion.__class__.__name__,
        total_time
    ])
