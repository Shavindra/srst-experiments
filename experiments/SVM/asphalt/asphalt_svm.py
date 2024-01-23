from sklearn import svm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import joblib

import sys
sys.path.append('../../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader

from utils import svm_loader as dl
import torch

class_name = 'asphalt'
image_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{class_name}'
label_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{class_name}'
val_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{class_name}'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf = svm.SVC()
from sklearn.metrics import jaccard_score

# Initialize the data loaders
train_loader = dl.SRST_DataloaderSVM(mask_dir=label_dir, image_dir=image_dir)
test_loader = dl.SRST_DataloaderSVM(mask_dir=val_dir, image_dir=image_dir)

# Initialize lists to store the images and labels
X_train = []
y_train = []
X_test = []
y_test = []

# Iterate over the train loader
for images, labels in train_loader.data_loader:
    for image, label in zip(images, labels):
        X_train.append(image.to(device))
        y_train.append(label.to(device))

# Iterate over the test loader
for images, labels in test_loader.data_loader:
    for image, label in zip(images, labels):
        X_test.append(image.to(device))
        y_test.append(label.to(device))

# Convert the lists to numpy arrays
X_train = torch.cat(X_train, dim=0).cpu().numpy()
y_train = torch.cat(y_train, dim=0).cpu().numpy()
X_test = torch.cat(X_test, dim=0).cpu().numpy()
y_test = torch.cat(y_test, dim=0).cpu().numpy()

# Now you can fit the SVM
clf.fit(X_train, y_train)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/svm_training')

# Make predictions on the train and test sets
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Calculate Jaccard similarity
jaccard_train = jaccard_score(y_train, y_pred_train, average='binary', pos_label=255)
jaccard_test = jaccard_score(y_test, y_pred_test, average='binary', pos_label=255)

# Log the Jaccard similarity
writer.add_scalar('SVM/Training Jaccard', jaccard_train)
writer.add_scalar('SVM/Testing Jaccard', jaccard_test)

# Log the training accuracy
train_accuracy = clf.score(X_train, y_train)
writer.add_scalar('SVM/Training Accuracy', train_accuracy)

# Log the testing accuracy
test_accuracy = clf.score(X_test, y_test)
writer.add_scalar('SVM/Testing Accuracy', test_accuracy)

writer.close()

results_dir = '/home/sfonseka/dev/SRST/srst-dataloader/experiments/SVM'

EXP_NAME = f'{class_name}_svm'
EXP_VERSION = 1
joblib.dump(clf, f'/models/{class_name}_svm.pkl')

print('Results:')
print(f'Class Name: {class_name}')
print(f'{EXP_NAME} Version: {EXP_VERSION}')
print(f'Jaccard Similarity (Train): {jaccard_train}')
print(f'Jaccard Similarity (Test): {jaccard_test}')
print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')


with open(f'{results_dir}/{EXP_NAME}.csv', 'a') as f:
    f.write(f'{EXP_NAME},{EXP_VERSION},{class_name},{jaccard_train},{jaccard_test},{train_accuracy},{test_accuracy}\n')


print('Finished Training')

# Save the model
