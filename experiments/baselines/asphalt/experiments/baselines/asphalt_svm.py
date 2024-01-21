from sklearn import svm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import svm_loader  as dl

class_name = 'asphalt'
image_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{class_name}'
label_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/train/512/{class_name}'
val_dir = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/val/512/{class_name}'

clf = svm.SVC()

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
        X_train.append(image)
        y_train.append(label)

# Iterate over the test loader
for images, labels in test_loader.data_loader:
    for image, label in zip(images, labels):
        X_test.append(image)
        y_test.append(label)

# Convert the lists to numpy arrays
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Now you can fit the SVM
clf.fit(X_train, y_train)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/svm_training')

# Log the training accuracy
train_accuracy = clf.score(X_train, y_train)
writer.add_scalar('Training Accuracy', train_accuracy)

# Log the testing accuracy
test_accuracy = clf.score(X_test, y_test)
writer.add_scalar('Testing Accuracy', test_accuracy)

writer.close()

print('Finished Training')