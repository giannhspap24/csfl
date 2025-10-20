#=====================================================
# Centralized (normal) learning: ResNet18 on HAM10000
# Single program
# ====================================================
import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
from torchvision import datasets, transforms
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

    
#===================================================================    
program = "Normal Learning AlexNet on MNIST"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset download and preparation
train_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
                    ])

test_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])

dataset_train = datasets.MNIST(root='./data', train=True, transform=train_transforms, download=True)
dataset_test = datasets.MNIST(root='./data', train=False, transform=test_transforms, download=True)

train_iterator = DataLoader(dataset_train, shuffle=True, batch_size=32)
test_iterator = DataLoader(dataset_test, batch_size=32)

print(f'Number of training examples: {len(dataset_train)}')
print(f'Number of testing examples: {len(dataset_test)}')
#=============================================================================
#                    Model definition: AlexNet
#============================================================================= 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 4 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the flattened size for FC layers
        # For MNIST (28x28):
        # - After conv1 + pool: (28 / 2) = 14
        # - After conv2 + pool: (14 / 2) = 7
        # - After conv3 + pool: (7 / 2) = 3.5 -> floor to 3
        # - After conv4 + pool: (3 / 2) = 1 (rounded down)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)  # Adjusted flattened size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        #x = self.pool(x)
        # Flatten the output of the last conv layer to feed into the FC layer
        x = x.view(x.size(0), -1)  # Flatten the feature map
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x



#net_glob = ResNet18(BasicBlock, [2, 2, 2, 2], 7) #7 is my numbr of classes
net_glob = SimpleCNN(num_classes=10)

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)   # to use the multiple GPUs 

net_glob.to(device)
print(net_glob)        


#=============================================================================
#                    Model Training and Testing
#============================================================================= 

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

#==========================================================================================================================     
def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    ell = len(iterator)
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad() # initialize gradients to zero
        
        # ------------- Forward propagation ----------
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy (fx , y)
        
        # -------- Backward propagation -----------
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / ell, epoch_acc / ell
        
def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    ell = len(iterator)
    
    with torch.no_grad():
        for (x,y) in iterator:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            fx = model(x)       
            loss = criterion(fx, y)
            acc = calculate_accuracy (fx , y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss/ell, epoch_acc/ell
 

# =======================================================================================
epochs = 200
LEARNING_RATE = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_glob.parameters(), lr = LEARNING_RATE)

loss_train_collect = []
loss_test_collect = []
acc_train_collect = []
acc_test_collect = []
        
start_time = time.time()    
for epoch in range(epochs):
    train_loss, train_acc = train(net_glob, device, train_iterator, optimizer, criterion)
    #print(f'Train completed - {epoch} Epoch")
    test_loss, test_acc = evaluate(net_glob, device, test_iterator, criterion)
    #print(f'Test completed - {epoch} Epoch")
    
    loss_train_collect.append(train_loss)
    loss_test_collect.append(test_loss)
    acc_train_collect.append(train_acc)
    acc_test_collect.append(test_acc)
    
    
    print(f'Train => Epoch: {epoch} \t Acc: {train_acc*100:05.2f}% \t Loss: {train_loss:.3f}')
    print(f'Test =>               \t Acc: {test_acc*100:05.2f}% \t Loss: {test_loss:.3f}')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')
#===================================================================================     
print("Training and Evaluation completed!")    
#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)