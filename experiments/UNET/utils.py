# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append('../../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader

# Now you can import your module
from models.UNET import UNetBaseline
from utils import dataloader as dl

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryROC, BinaryPrecisionRecallCurve


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

CLASS_NAME = 'asphalt'
EXPERIMENT_MODEL = 'UNET'
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

# %%



# %%
from tqdm import tqdm

# [Your import statements and other code]

#writer = TensorBoardLogger(TENSOIRBOARD_DIR, name=EXPERIMENT_NAME)

writer = SummaryWriter(TENSOIRBOARD_DIR)

# Create a CSV file and write the headers
MODEL_RESULT_FILE = f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/{EXPERIMENT_MODEL}/experiment_results.csv'
lock = FileLock("/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/experiment_results.lock")
with open(f'{MODEL_RESULT_FILE}', 'a', newline='') as file:
    exp_stats = csv.writer(file)
#    exp_stats.writerow(["Experiment", "Version", "Epoch", "Train Loss", "Validation Loss", "Train IoU", "Validation IoU", "Train Accuracy", "Validation Accuracy"])


best_loss = float('inf')
best_iou = 0


EPOCHS = 20
THRESHOLD = 0.5  # Adjust as needed
MASK_COUNT = 20

LR = 0.01



epoch_number = 0
best_loss = float('inf')
best_iou = 0

metric_iou = IoU().to(DEVICE)  # Initialize IoU metric for binary classification
metric_accuracy = BinaryAccuracy().to(DEVICE)  # Initialize accuracy metric for binary classification
metric_auroc = BinaryAUROC().to(DEVICE)  # Initialize AUROC metric for binary classification
metric_precision = BinaryPrecision().to(DEVICE)  # Initialize precision metric for binary classification
metric_recall = BinaryRecall().to(DEVICE)  # Initialize recall metric for binary classification
metric_f1 = BinaryF1Score().to(DEVICE)  # Initialize F1 metric for binary classification
metric_roc = BinaryPrecisionRecallCurve().to(DEVICE)  # Initialize ROC metric for binary classification


new_metrics = [
    {'name': 'iou', 'metric': metric_iou},
    {'name': 'accuracy', 'metric': metric_accuracy},
    {'name': 'auroc', 'metric': metric_auroc},
        {'name': 'precision', 'metric': metric_precision},
        {'name': 'recall', 'metric': metric_recall},
        {'name': 'f1', 'metric': metric_f1},
        # {'name': 'roc', 'metric': metric_roc},
    ]

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    print('Training')
    model.train()
    total_loss = 0
    
        # Reset metrics at the start of the epoch
    for metric in new_metrics:
        metric['metric'].reset()


    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, masks, ___path in progress_bar:
        images, masks = images.to(device), masks.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = model(images)
        outputs_sigmoid = torch.sigmoid(outputs)  # Apply sigmoid to convert to probabilities

        # Compute the loss and its gradients
        loss = criterion(outputs, masks)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        total_loss += loss.item()

        # Update IoU metric
        # VERIFY THIS 
        preds = (outputs_sigmoid > THRESHOLD).int()  # Convert outputs to binary predictions

        outputs_sigmoid_cpu = outputs_sigmoid
        preds_cpu = preds
        masks_cpu = masks

        # Update metrics
        for metric in new_metrics:
            if metric['name'] == 'iou' or metric['name'] == 'accuracy':
                metric['metric'].update(preds_cpu, masks_cpu.int())
            else:
                metric['metric'].update(outputs_sigmoid_cpu, masks_cpu.int())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    new_metric_scores = [{'metric_name': 'train_loss', 'metric_score': avg_loss}]
    
    for metric in new_metrics:
        new_metric_scores.append({
            'metric_name': f'train_{metric["name"]}', 
            'metric_score': metric['metric'].compute()
        })


    # print('EPOCH METRICS', new_metric_scores)
    return new_metric_scores



metric_eval_iou = IoU().to(DEVICE)  # Initialize IoU metric for binary classification
metric_eval_accuracy = BinaryAccuracy().to(DEVICE)  # Initialize accuracy metric for binary classification
metric_eval_auroc = BinaryAUROC().to(DEVICE)  # Initialize AUROC metric for binary classification
metric_eval_precision = BinaryPrecision().to(DEVICE)  # Initialize precision metric for binary classification
metric_eval_recall = BinaryRecall().to(DEVICE)  # Initialize recall metric for binary classification
metric_eval_f1 = BinaryF1Score().to(DEVICE)  # Initialize F1 metric for binary classification
metric_eval_roc = BinaryPrecisionRecallCurve().to(DEVICE)  # Initialize ROC metric for binary classification

new_eval_metrics = [
    {'name': 'iou', 'metric': metric_eval_iou},
    {'name': 'accuracy', 'metric': metric_eval_accuracy},
    {'name': 'auroc', 'metric': metric_eval_auroc},
    {'name': 'precision', 'metric': metric_eval_precision},
    {'name': 'recall', 'metric': metric_eval_recall},
    {'name': 'f1', 'metric': metric_eval_f1},
    # {'name': 'roc', 'metric': metric_eval_roc},
]

def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    # Reset metrics at the start of the epoch
    for metric in new_eval_metrics:
        metric['metric'].reset()

    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for eval_images, eval_masks, __p in progress_bar:
            eval_images, eval_masks = eval_images.to(device), eval_masks.to(device)
            outputs = model(eval_images)
            loss = criterion(outputs, eval_masks)
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            outputs_sigmoid = torch.sigmoid(outputs)


          # Update IoU metric
            # For binary classification, you can use a threshold to convert outputs to binary format
            eval_preds = (outputs_sigmoid > THRESHOLD).int()  # Adjust THRESHOLD as needed, e.g., 0.5
            
            outputs_sigmoid_cpu = outputs_sigmoid
            eval_preds_cpu = eval_preds
            eval_masks_cpu = eval_masks

        # Update metrics
        for metric in new_eval_metrics:
            if metric['name'] == 'iou' or metric['name'] == 'accuracy':
                metric['metric'].update(eval_preds_cpu, eval_masks_cpu.int())
            else:
                metric['metric'].update(outputs_sigmoid_cpu, eval_masks_cpu.int())

            progress_bar.set_postfix(loss=loss.item())

    avg_eval_loss = total_loss / len(val_loader)
    new_eval_metric_scores = [{'metric_name': 'eval_loss', 'metric_score': avg_eval_loss}]

    for metric in new_eval_metrics:
        new_eval_metric_scores.append({
            'metric_name': f'eval_{metric["name"]}',
            'metric_score': metric['metric'].compute()
        })

 #   print('EVAL METRICS', new_eval_metric_scores)
    return new_eval_metric_scores

import time

# Record the start time
start_time = time.time()
patience = 15  # Number of epochs to wait for improvement before stopping
best_score = 0  # Best score achieved so far
wait = 0  # Number of epochs we have waited so far without improvement


import matplotlib.pyplot as plt
import io

def log_pr_curve(writer, precision, recall, epoch, phase):
    # Move tensors to CPU if they are on GPU
    precision = precision.cpu()
    recall = recall.cpu()

    # Create the PR curve plot
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{phase} Precision-Recall Curve - Epoch {epoch}')
    ax.legend()
    ax.grid(True)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_tensor = plt.imread(buf, format='png')
    image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)  # Convert to CxHxW

    writer.add_image(f'{phase}/Precision-Recall Curve', image_tensor, global_step=epoch)
    # Close the plot
    plt.close(fig)


for epoch in tqdm(range(EPOCHS), desc='Epochs'):  # tqdm wrapper for epochs
    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_metrics = eval_model(model, val_loader, criterion, DEVICE)

    logging_step = epoch + 1  # assuming epoch starts from 0
    val_metric_iou = 0

    metrics_dict = [
        EXPERIMENT_NAME,
        EXPERIMENT_NAME_VERSION,
        epoch
    ]

    for i, val_metric in enumerate(val_metrics):
        metric_name = val_metric['metric_name'].split('_')[1]
        val_metric_score = val_metric['metric_score']
        train_metric_score = train_metrics[i]['metric_score']

        if metric_name == 'roc':
                # Log metrics
            # Compute metrics
            train_precision, train_recall, train_thresholds = train_metric_score
            val_precision, val_recall, val_thresholds = val_metric_score

            # Log metrics
            log_pr_curve(writer, train_precision, train_recall, epoch, 'Train')
            log_pr_curve(writer, val_precision, val_recall, epoch, 'Validation')

            # writer.add_scalar(f'Training/{metric_name}/', train_metric_score, logging_step, walltime=now_before)
            # writer.add_scalar(f'Validation/{metric_name}', val_metric_score,  logging_step, walltime=now_before)

            continue

        if hasattr(val_metric_score, 'item'):
            val_metric_score = val_metric_score.item()

        # Find corresponding metric in val_metrics
        if val_metric_score and hasattr(val_metric_score, 'item'):
            val_metric_score = val_metric_score.item()
            train_metric_score = train_metric_score.item()

        print(f'Epoch {epoch}, Logging Step: {logging_step}, {metric_name} - Train: {train_metric_score}, Val: {val_metric_score}')

        print(f'Logging to Tensorboard: {metric_name} - Train: {train_metric_score}, Val: {val_metric_score}')

        writer.add_scalar(f'Training/{metric_name}', train_metric_score, logging_step, walltime=now_before)
        writer.add_scalar(f'Validation/{metric_name}', val_metric_score,  logging_step, walltime=now_before)
        writer.add_scalars(f'Training vs Validation - {metric_name}', {'train': train_metric_score, 'val': val_metric_score},  logging_step, walltime=now_before)

        metrics_dict.append(train_metric_score)
        metrics_dict.append(val_metric_score)

    epoch += 1  # Increment epoch number for the next loop

    # Write the metrics for this epoch to the CSV file
    with lock:
        with open(f'{MODEL_RESULT_FILE}', 'a', newline='') as file:
            exp_stats = csv.writer(file)
            exp_stats.writerow(metrics_dict)

            print('Wrote metrics to CSV file: ', f'{MODEL_RESULT_FILE}')


    writer.flush()
    # Save the model if it's the best one so far
    # Save the model if it's the best one so far in terms of IoU
    if val_metric_iou > best_iou:
        best_iou = val_metric_iou
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'best_model_{EXPERIMENT_NAME_VERSION}.pt'))
        print(f'Saved new best model with IoU {best_iou:.4f} at epoch {epoch_number}')
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
