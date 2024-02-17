#from sklearn import svm
from thundersvm import SVC

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

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

    from datetime import datetime, timedelta
    now = datetime.now()
    now_before = datetime.now() - timedelta(days=14)
    now_before = now_before.timestamp()
    now = now_before

    EXP_NAME = f'{class_name}_svm'
    EXP_VERSION = 1
    results_dir = '/home/sfonseka/dev/SRST/srst-dataloader/experiments/SVM'

    CLASS_NAME = class_name
    EXPERIMENT_MODEL = 'SVM'
    DATASET_VARIANT = 'binary_grayscale'
    EXPERIMENT_NAME= f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}'

    EXPERIMENT_NAME= f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}'
    EXPERIMENT_NAME_VERSION = f'{EXPERIMENT_MODEL}_{DATASET_VARIANT}_{CLASS_NAME}_{now}'

    RESULT_DIR = f'runs/{EXPERIMENT_NAME_VERSION}'
    LOG_DIR = f'runs/{EXPERIMENT_NAME_VERSION}/logs'
    MODEL_SAVE_PATH = f'runs/{EXPERIMENT_NAME_VERSION}/models'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(LOG_DIR)
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize the SVM
    clf = SVC( kernel='rbf', C=1.0, random_state=42, verbose=True, n_jobs=40, max_iter=50)
    print('n_jobs: ', clf.n_jobs)
    print('tol: ', clf.tol)
    print(clf)


    # Initialize the data loaders
    train_loader = dl.SRST_DataloaderSVM(mask_dir=label_dir, image_dir=image_dir, mask_size=500)
    val_loader = dl.SRST_DataloaderSVM(mask_dir=val_dir, image_dir=image_dir, mask_size=500)
    test_loader = dl.SRST_DataloaderSVM(mask_dir=test_dir, image_dir=image_dir, mask_size=500)   

    # Initialize lists to store the images and labels
    img_train = []
    label_train = []
    img_val = []
    label_val = []

    # Iterate over the train loader
    for images, labels in train_loader.data_loader:
        for image, label in zip(images, labels):
            # print(image.shape, label.shape)
            img_train.append(image.to(device))
            label_train.append(label.to(device))

    # Iterate over the val loader
    for images, labels in val_loader.data_loader:
        for image, label in zip(images, labels):
            # print(image.shape, label.shape)
            img_val.append(image.to(device))
            label_val.append(label.to(device))

    for img, lb in test_loader.data_loader:
        for im, l in zip(img, lb):
            img_test = im.to(device)
            label_test = l.to(device)

    # Convert the lists to numpy arrays
    img_train = torch.cat(img_train, dim=0).cpu().numpy()
    label_train = torch.cat(label_train, dim=0).cpu().numpy()
    img_val = torch.cat(img_val, dim=0).cpu().numpy()
    label_val = torch.cat(label_val, dim=0).cpu().numpy()

    img_test = img_test.cpu().numpy()
    label_test = label_test.cpu().numpy()


    # Now you can fit the SVM
    print('Fitting the SVM')
    clf.fit(img_train, label_train)

    # Make predictions on the train and val sets
    print('Predicting on trainx sets')
    y_pred_train = clf.predict(img_train)

    print('Predicting on val set')
    y_pred_val = clf.predict(img_val)

    print('Predicting on test set')
    y_pred_test = clf.predict(img_test)


    # Calculate Jaccard similarity
    print('y_train', y_pred_train)
    print('y_val', y_pred_val)
    print('y_test', y_pred_test)

    jaccard_train = jaccard_score(label_train, y_pred_train, average='binary', pos_label=255)
    jaccard_val = jaccard_score(label_val, y_pred_val, average='binary', pos_label=255)
    jaccard_test = jaccard_score(label_test, y_pred_test, average='binary', pos_label=255)

    # Log the Jaccard similarity
    writer.add_scalar('SVM/Training Jaccard', jaccard_train)
    writer.add_scalar('SVM/valing Jaccard', jaccard_val)
    writer.add_scalar('SVM/Test Jaccard', jaccard_test)

    # Log the training accuracy
    train_accuracy = clf.score(img_train, label_train)
    writer.add_scalar('SVM/Training Accuracy', train_accuracy)

    # Log the valing accuracy
    val_accuracy = clf.score(img_val, label_val)
    writer.add_scalar('SVM/valing Accuracy', val_accuracy)


    test_accuracy = clf.score(img_test, label_test)
    writer.add_scalar('SVM/Test Accuracy', test_accuracy)

    writer.close()

    print('Results:')
    print(f'Class Name: {class_name}')
    print(f'{EXP_NAME} Version: {EXP_VERSION}')
    print(f'Jaccard Similarity (Train): {jaccard_train}')
    print(f'Jaccard Similarity (val): {jaccard_val}')
    print(f'Training Accuracy: {train_accuracy}')
    print(f'valing Accuracy: {val_accuracy}')
    print(f'Test Accuracy: {clf.score(img_test, label_test)}')

    with open(f'{results_dir}/{EXP_NAME}.csv', 'a') as f:
        f.write(f'{EXP_NAME},{EXP_VERSION},{class_name},{jaccard_train},{jaccard_val},{train_accuracy},{val_accuracy},{jaccard_test}, {test_accuracy}\n')

    with open(f'{results_dir}/svm_results.csv', 'a') as f:
        f.write(f'{EXP_NAME},{EXP_VERSION},{class_name},{jaccard_train},{jaccard_val},{train_accuracy},{val_accuracy},{jaccard_test}, {test_accuracy}\n')


    clf.save_to_file(os.path.join(MODEL_SAVE_PATH, f'{class_name}_svm'))

    print('Finished Training')



# Save the model