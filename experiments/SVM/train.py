from thundersvm import SVC
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, jaccard_score
import sys

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append('../../../')
sys.path.append('../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader
sys.path.append('..')  # This is /home/sfonseka/dev/SRST/srst-dataloader/experiments/SVM

from utils import svm_loader as dl
import torch
import os

def train_svm(class_name):
    image_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{class_name}'
    label_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{class_name}'
    val_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{class_name}'
    test_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/test/512/{class_name}'

    now = datetime.now() - timedelta(days=14)
    timestamp = now.timestamp()

    EXP_NAME = f'{class_name}_svm_2'
    results_dir = '/home/sfonseka/dev/SRST/srst-dataloader/experiments/SVM'
    MODEL_SAVE_PATH = f'runs/{EXP_NAME}_{timestamp}/models'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    writer = SummaryWriter(f'runs/{EXP_NAME}_{timestamp}/logs')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = SVC(kernel='rbf', C=1.0, random_state=42, verbose=True, n_jobs=40, max_iter=50)

    train_loader = dl.SRST_DataloaderSVM(mask_dir=label_dir, image_dir=image_dir, mask_size=500)
    val_loader = dl.SRST_DataloaderSVM(mask_dir=val_dir, image_dir=image_dir, mask_size=500)
    test_loader = dl.SRST_DataloaderSVM(mask_dir=test_dir, image_dir=image_dir, mask_size=500)

    def load_data(loader):
        imgs, labels = [], []
        for images, labels_batch in loader.data_loader:
            imgs.extend([image.to(device) for image in images])
            labels.extend([label.to(device) for label in labels_batch])
        return torch.cat(imgs, dim=0).cpu().numpy(), torch.cat(labels, dim=0).cpu().numpy()

    img_train, label_train = load_data(train_loader)
    img_val, label_val = load_data(val_loader)
    img_test, label_test = load_data(test_loader)

    clf.fit(img_train, label_train)

    def compute_metrics(labels, predictions, dataset_type):
        precision = precision_score(labels, predictions, average='binary', pos_label=255)
        recall = recall_score(labels, predictions, average='binary', pos_label=255)
        f1 = f1_score(labels, predictions, average='binary', pos_label=255)
        jaccard = jaccard_score(labels, predictions, average='binary', pos_label=255)

        writer.add_scalar(f'SVM/{dataset_type} Precision', precision)
        writer.add_scalar(f'SVM/{dataset_type} Recall', recall)
        writer.add_scalar(f'SVM/{dataset_type} F1-Score', f1)
        writer.add_scalar(f'SVM/{dataset_type} Jaccard', jaccard)

        return precision, recall, f1, jaccard

    y_pred_train = clf.predict(img_train)
    y_pred_val = clf.predict(img_val)
    y_pred_test = clf.predict(img_test)

    train_metrics = compute_metrics(label_train, y_pred_train, 'Training')
    val_metrics = compute_metrics(label_val, y_pred_val, 'Validation')
    test_metrics = compute_metrics(label_test, y_pred_test, 'Test')

    train_accuracy = clf.score(img_train, label_train)
    val_accuracy = clf.score(img_val, label_val)
    test_accuracy = clf.score(img_test, label_test)

    writer.add_scalar('SVM/Training Accuracy', train_accuracy)
    writer.add_scalar('SVM/Validation Accuracy', val_accuracy)
    writer.add_scalar('SVM/Test Accuracy', test_accuracy)

    print('Results:')
    print(f'Class Name: {class_name}')
    print(f'{EXP_NAME} Metrics:')
    print(f'Training - Precision: {train_metrics[0]}, Recall: {train_metrics[1]}, F1-Score: {train_metrics[2]}, Jaccard: {train_metrics[3]}')
    print(f'Validation - Precision: {val_metrics[0]}, Recall: {val_metrics[1]}, F1-Score: {val_metrics[2]}, Jaccard: {val_metrics[3]}')
    print(f'Test - Precision: {test_metrics[0]}, Recall: {test_metrics[1]}, F1-Score: {test_metrics[2]}, Jaccard: {test_metrics[3]}')
    print(f'Accuracy - Training: {train_accuracy}, Validation: {val_accuracy}, Test: {test_accuracy}')
    print("Test Confusion Matrix:")
    print(confusion_matrix(label_test, y_pred_test))

    results = [EXP_NAME, class_name, train_metrics, val_metrics, test_metrics, train_accuracy, val_accuracy, test_accuracy]
    with open(f'{results_dir}/{EXP_NAME}.csv', 'a') as f:
        f.write(','.join(map(str, results)) + '\n')

    clf.save_to_file(os.path.join(MODEL_SAVE_PATH, f'{class_name}_svm'))
    writer.close()
    print('Finished Training')
