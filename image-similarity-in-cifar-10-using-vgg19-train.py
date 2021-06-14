import sys
print("Python %s" % sys.version)
import os, time
import pickle
import argparse

import numpy as np
print("NumPy %s" % np.__version__)

import PIL
print("PIL %s" % PIL.__version__)
from PIL import Image

import torch
print("PyTorch %s" % torch.__version__)
from torch.utils.data import Dataset, DataLoader

import torchvision
print("torchvision %s" % torchvision.__version__)
from torchvision import datasets, transforms, models

### Hyperparameters
# Data
INPUT_DIRECTORY_NAME = '/datastores'
NUM_CLASSES = 10
WORKING_DIRECTORY = '/work'

# Training
BATCH_SIZE = 100
NUM_WORKERS = 2 #bar 12
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 2 #bar 100

parser = argparse.ArgumentParser(description='Image Similarity Training')
parser.add_argument('-LR', '--LEARNING_RATE', type=float,
                    help='Learning rate.')
args = parser.parse_args()

if LEARNING_RATE in args:
    LEARNING_RATE = args.LEARNING_RATE

print('INPUT_DIRECTORY_NAME', INPUT_DIRECTORY_NAME)
print('NUM_CLASSES', NUM_CLASSES)
print('WORKING_DIRECTORY', WORKING_DIRECTORY)
print('BATCH_SIZE', BATCH_SIZE)
print('NUM_WORKERS', NUM_WORKERS)
print('LEARNING_RATE', LEARNING_RATE)
print('MOMENTUM', MOMENTUM)
print('WEIGHT_DECAY', WEIGHT_DECAY)
print('NUM_EPOCHS', NUM_EPOCHS)

asdfasdfasdfasdf

### Data Analysis
for dirname, _, filenames in os.walk(INPUT_DIRECTORY_NAME):
    for filename in filenames:
        print(os.path.join(dirname, filename))

input_dir_name = INPUT_DIRECTORY_NAME

cifar_10_dir_name = os.path.join(input_dir_name,
                                 'the-cifar10-dataset')
print(cifar_10_dir_name)

train1_full_filename = os.path.join(cifar_10_dir_name, 'data_batch_1')
train2_full_filename = os.path.join(cifar_10_dir_name, 'data_batch_2')
train3_full_filename = os.path.join(cifar_10_dir_name, 'data_batch_3')
train4_full_filename = os.path.join(cifar_10_dir_name, 'data_batch_4')
train5_full_filename = os.path.join(cifar_10_dir_name, 'data_batch_5')
test_full_filename = os.path.join(cifar_10_dir_name, 'test_batch')
meta_full_filename = os.path.join(cifar_10_dir_name, 'batches.meta')

print(train1_full_filename)
print(train2_full_filename)
print(train3_full_filename)
print(train4_full_filename)
print(train5_full_filename)
print(test_full_filename)
print(meta_full_filename)

def unpickle_meta(full_filename):
    with open(full_filename, 'rb') as fh:
        meta = pickle.load(fh, encoding='bytes')
    return meta

meta = unpickle_meta(meta_full_filename)
print(meta)

class_index_to_name = {}
for i, class_name in enumerate(meta[b'label_names']):
    class_index_to_name[i] = class_name.decode('ascii')
    
print(class_index_to_name)

assert(len(class_index_to_name) == NUM_CLASSES)

def unpickle_input(full_filename):
    with open(full_filename, 'rb') as fh:
        images_dict = pickle.load(fh, encoding='bytes')
    return images_dict

train1 = unpickle_input(train1_full_filename)
train2 = unpickle_input(train2_full_filename)
train3 = unpickle_input(train3_full_filename)
train4 = unpickle_input(train4_full_filename)
train5 = unpickle_input(train5_full_filename)
test = unpickle_input(test_full_filename)

print(type(train1))

print(type(train1[b'batch_label']))

print(train1[b'batch_label'])
print(train2[b'batch_label'])
print(train3[b'batch_label'])
print(train4[b'batch_label'])
print(train5[b'batch_label'])
print(test[b'batch_label'])

print(type(train1[b'labels']))

print('Train 1:', len(train1[b'labels']), 'images')
print('Train 2:', len(train2[b'labels']), 'images')
print('Train 3:', len(train3[b'labels']), 'images')
print('Train 4:', len(train4[b'labels']), 'images')
print('Train 5:', len(train5[b'labels']), 'images')
print('Test 1 :', len(test[b'labels']), 'images')

print(train1[b'labels'][:10])

print(type(train1[b'data']))

print(train1[b'data'].shape)

print(train1[b'data'][0])

print(type(train1[b'filenames']))

print(type(train1[b'filenames'][0]))

print(train1[b'filenames'][:10])

### Data Preprocessing
class CifarDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # Convert to PIL image for consistency.
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


train_images = []
train_images.append(train1[b'data'])
train_images.append(train2[b'data'])
train_images.append(train3[b'data'])
train_images.append(train4[b'data'])
train_images.append(train5[b'data'])

test_images = []
test_images.append(test[b'data'])

train_images = np.vstack(train_images) \
    .reshape(-1, 3, 32, 32) \
    .transpose((0, 2, 3, 1))

print(train_images.shape)

test_images = np.vstack(test_images) \
    .reshape(-1, 3, 32, 32) \
    .transpose((0, 2, 3, 1))

print(test_images.shape)

train_labels = []
train_labels.extend(train1[b'labels'])
train_labels.extend(train2[b'labels'])
train_labels.extend(train3[b'labels'])
train_labels.extend(train4[b'labels'])
train_labels.extend(train5[b'labels'])

print(len(train_labels))

print(train_labels[:10])

test_labels = []
test_labels.extend(test[b'labels'])

print(len(test_labels))

assert(train_images.shape[0] == len(train_labels))

assert(test_images.shape[0] == len(test_labels))

train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
    ])

test_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
    ])

train_datasets = CifarDataset(train_images, train_labels, train_transforms)
test_datasets = CifarDataset(test_images, test_labels, test_transforms)

print(len(train_datasets))

print(len(train_datasets))

print(len(test_datasets))

print(train_datasets[0])

print(test_datasets[0])

train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS, shuffle=True)

print(train_loader)

print(test_loader)

vgg = models.vgg19_bn(num_classes=NUM_CLASSES, pretrained=False)

print(vgg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

vgg.to(device)

# Categorical cross-entropy
criterion = torch.nn.CrossEntropyLoss()

# Stochastic gradient descent with momentum. Learn only the classifier layers.
optimizer = torch.optim.SGD(vgg.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY)

model_directory = os.path.join(WORKING_DIRECTORY, 'models')
print(model_directory)

if not os.path.isdir(model_directory):
    print(f'Creating {model_directory} directory...')
    os.makedirs(model_directory)

checkpoint_full_filename = os.path.join(model_directory, 'checkpoint.pyt')
print(checkpoint_full_filename)

def save_checkpoint(model, optimizer, filename='checkpoint.pyt'):
    checkpoint = {'model_state_dict': model.state_dict(),
                  'model_output_class': model.classifier[6].out_features,
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, filename)

### Model Training

best_test_accuracy = 0.0
train_losses = []
test_losses = []
test_accuracies = []

for epoch_i in range(1, NUM_EPOCHS+1):
    # Train portion
    ###############
    tic = time.time()
    batch_train_loss = 0.0
    batch_train_loss_counter = 0
    
    # Suitch to train mode
    vgg.train()
    for batch_i, (data, target) in enumerate(train_loader):
        # Move data to running device
        data, target = data.to(device), target.to(device)
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass
        output = vgg(data)
        # Calculate batch loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update parameter
        optimizer.step()
        # Save last training loss
        batch_train_loss += loss.item()
        batch_train_loss_counter += 1
        
    train_loss = batch_train_loss / batch_train_loss_counter
    
    toc = time.time()
    train_time = toc - tic
    
    # Test portion
    ##############
    tic = time.time()
    # Track test loss over 10 classes
    test_loss = 0.0
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    
    # Switch to evaluation (inference) mode
    vgg.eval()
    for data, target in test_loader:
        # Move data to running device
        data, target = data.to(device), target.to(device)
        # Forward pass
        output = vgg(data)
        # Calculate batch loss
        loss = criterion(output, target)
        # Update test loss 
        test_loss += loss.item() * data.size(0)
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # Compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # Calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = (100. * np.sum(class_correct)) / np.sum(class_total)
    
    toc = time.time()
    test_time = toc - tic
    
    # Summary portion
    #################
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # TrTm: Training Time
    # TrLs: Training Loss
    # TeTm: Test Time
    # TeLs: Test Loss
    # TeAc: Test Accuracy
    summary = ("%3d, " % epoch_i) + \
              ("TrTm %3ds, " % int(train_time)) + \
              ('TrLs {:6.4f}, '.format(train_loss)) + \
              ("TeTm %3ds, " % int(test_time)) + \
              ('TeLs {:6.4f}, '.format(test_loss)) + \
              ('TeAc %3d%%' % (test_accuracy))
    print(summary)

    # Save model checkpoint if test accuracy improves
    if test_accuracy > best_test_accuracy:
        print('Saving Checkpoint...')
        save_checkpoint(vgg, optimizer, filename=checkpoint_full_filename)
        best_test_accuracy = test_accuracy

