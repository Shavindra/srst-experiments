# %%

import sys
import os
import time

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
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryAccuracy, BinaryPrecision as Precision, BinaryRecall as Recall, BinaryF1Score as F1


# LOGGING
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
import csv
from filelock import FileLock

torch.manual_seed(42)


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

EXPERIMENT_MODEL = 'UNET'
DATASET_VARIANT = 'binary_grayscale'


def train_unet(class_name, epochs=3, threshold=0.5, mask_count=6, learning_rate=0.001):


    def save_metrics_to_csv(csv_file_path, experiment_name, experiment_version, epoch, train_metrics, val_metrics, lock=None):
        # Define field names for CSV file
        field_names = ['Experiment_Name', 'Experiment_Version', 'Epoch'] + \
                    [f'train_{m}' for m in train_metrics.keys()] + \
                    [f'val_{m}' for m in val_metrics.keys()]

            # Check if the file exists and has content
        file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0

        # Acquire lock if provided (useful in multi-threaded environments)
        if lock:
            lock.acquire()

        try:
            with open(csv_file_path, 'a', newline='') as file:
                exp_stats = csv.writer(file)

                # Write header if the file doesn't exist or is empty
                if not file_exists:
                    exp_stats.writerow(field_names)

                # Write data
                row = [experiment_name, experiment_version, epoch] + \
                    list(train_metrics.items()) + list(val_metrics.items())
                exp_stats.writerow(row)
                print('Wrote metrics to CSV file:', csv_file_path)
        finally:
            # Release lock if it was acquired
            if lock:
                lock.release()


    def initialize_metrics(ignore_index=None):
        metrics = {
            'iou': IoU(ignore_index=ignore_index).to(DEVICE),
            'accuracy': BinaryAccuracy(ignore_index=ignore_index).to(DEVICE),
            'precision': Precision(ignore_index=ignore_index).to(DEVICE),
            'recall': Recall(ignore_index=ignore_index).to(DEVICE),
            'f1': F1(ignore_index=ignore_index).to(DEVICE)
        }
        return metrics

    def update_metrics(metrics, preds, masks):
        for metric in metrics.values():
            metric.update(preds, masks)



    EPOCHS = epochs
    THRESHOLD = threshold  # Adjust as needed
    MASK_COUNT = mask_count
    LR = learning_rate

    CLASS_NAME = class_name
    IMG_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{CLASS_NAME}'
    LABEL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{CLASS_NAME}'
    VAL_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{CLASS_NAME}'

    EXPERIMENT_NAME= f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}'
    EXPERIMENT_NAME_VERSION = f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}_MASKED_METRICS_{now}'

    RESULT_DIR = f'runs/{EXPERIMENT_NAME_VERSION}'
    LOG_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/logs'
    TENSOIRBOARD_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/tensorboard'

    MODEL_SAVE_PATH = f'runs/{EXPERIMENT_NAME_VERSION}/models'

    # Create directories to save results
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


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


    dataloader = dl.SRST_DataloaderGray(mask_dir=LABEL_DIR, image_dir=IMG_DIR, mask_count=MASK_COUNT)
    val_dataloader = dl.SRST_DataloaderGray(mask_dir=VAL_DIR, image_dir=IMG_DIR, mask_count=MASK_COUNT)


    # Example of model instantiation
    model = UNetBaseline(out_classes=1).to(DEVICE)  # For grayscale, out_classes should be 1

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = dataloader.data_loader
    val_loader = val_dataloader.data_loader

    # %%
    print('MASK_COUNT: ', MASK_COUNT)
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


    epoch_number = 0
    best_loss = float('inf')
    best_iou = 0

    # Record the start time
    start_time = time.time()
    patience = 20  # Number of epochs to wait for improvement before stopping
    wait = 0  # Number of epochs we have waited so far without improvement

    best_val_loss = float('inf')

    best_mask_iou = 0

    def train_one_epoch(model, train_loader, criterion, optimizer, device):
        print('Training')
        model.train()
        total_loss = 0

        # Initialize metrics
        metrics = initialize_metrics()
        metrics_masked = initialize_metrics(ignore_index=0)

        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for images, masks, __paths in progress_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > THRESHOLD).int()

            # Update metrics
            update_metrics(metrics, preds, masks)
            update_metrics(metrics_masked, preds, masks)

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # Compute final metrics
        final_metrics = {k: metric.compute() for k, metric in metrics.items()}
        final_metrics_masked = {f"{k}_masked": metric.compute() for k, metric in metrics_masked.items()}
        
        final_metrics.update(final_metrics_masked)
        final_metrics['train_loss'] = avg_loss

        return final_metrics


    def eval_model(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0

        # Initialize metrics
        metrics = initialize_metrics()
        metrics_masked = initialize_metrics(ignore_index=0)

        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for eval_images, eval_masks, __p in progress_bar:
                eval_imgs, eval_msks = eval_images.to(device), eval_masks.to(device)
                outputs = model(eval_imgs)
                loss = criterion(outputs, eval_msks)
                total_loss += loss.item()

                eval_preds = (torch.sigmoid(outputs) > THRESHOLD).int()

                # Update metrics
                update_metrics(metrics, eval_preds, eval_msks)
                update_metrics(metrics_masked, eval_preds, eval_msks)

                progress_bar.set_postfix(loss=loss.item())

        avg_eval_loss = total_loss / len(val_loader)

        # Compute final metrics
        final_metrics = {k: metric.compute() for k, metric in metrics.items()}
        final_metrics_masked = {f"{k}_masked": metric.compute() for k, metric in metrics_masked.items()}
        
        final_metrics.update(final_metrics_masked)
        final_metrics['eval_loss'] = avg_eval_loss

        return final_metrics



    # %%

    # Save the model if it's the best one so far in terms of loss or IoU
    def save_model_checkpoint(model, optimizer, epoch, best_loss, best_iou, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_iou': best_iou
        }, filename)

        # Early stopping check...

    for epoch in tqdm(range(EPOCHS), desc='Epochs'):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = eval_model(model, val_loader, criterion, DEVICE)

        print(val_metrics)

        logging_step = epoch + 1

        # Print and log metrics
        for metric_name, value in {**train_metrics, **val_metrics}.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f'Epoch {epoch}, {metric_name}: {value}')
            writer.add_scalar(f'Metrics/{metric_name}', value, logging_step)

        # Check and save model based on best validation loss and IoU
        if val_metrics['eval_loss'] < best_val_loss:
            best_val_loss = val_metrics['eval_loss']
            save_model_checkpoint(model, optimizer, logging_step, best_val_loss, best_iou, f'best_loss_model_{EXPERIMENT_NAME_VERSION}.pt')

        if val_metrics['iou_masked'] > best_mask_iou:
            best_mask_iou = val_metrics['eval_masked_iou']
            save_model_checkpoint(model, optimizer, logging_step, best_val_loss, best_iou, f'best_mask_iou_model_{EXPERIMENT_NAME_VERSION}.pt')

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['eval_iou']
            save_model_checkpoint(model, optimizer, logging_step, best_val_loss, best_iou, f'best_iou_model_{EXPERIMENT_NAME_VERSION}.pt')
        else:
            wait += 1

        # Early stopping
        if wait >= patience:
            print("Early stopping")
            break

        # Usage in your training loop
        csv_file_path = f'{MODEL_RESULT_FILE}'

        for epoch in tqdm(range(EPOCHS), desc='Epochs'):

            # Save metrics to CSV
            save_metrics_to_csv(csv_file_path, EXPERIMENT_NAME, EXPERIMENT_NAME_VERSION, epoch + 1, train_metrics, val_metrics, lock)


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
