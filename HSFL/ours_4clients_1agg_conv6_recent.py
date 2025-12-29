import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import time
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from prepare_datasets import DatasetSplit
from prepare_datasets import SkinData
from torchvision import datasets, transforms

CHECKPOINT_DIR = "./client_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SERVER_CHECKPOINT_DIR = "./server_checkpoints"
os.makedirs(SERVER_CHECKPOINT_DIR, exist_ok=True)

def client_ckpt_path(idx):
    return os.path.join(CHECKPOINT_DIR, f"client_{idx}.pt")

def server_ckpt_path(idx):
    return os.path.join(SERVER_CHECKPOINT_DIR, f"server_{idx}.pt")

def set_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seeds(seed=1234)
#It needs fix, the accuracy is going up and down
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    
# Hyperparameters
num_users = 30
global_epochs = 2
frac = 1        # Participation of clients
lr = 0.0001
batch_size = 32
fedavg_freq=1
#Program identifier
cut_layer = "conv6"
collaborative_layer = "conv3"  # Starting conv layer for selective averagingmodel="VGG-11" 
program = "vgg11_CIFAR10_conv6_4clients_1agg_iid_test_checkpoint"
print(f"---------{program}----------")
start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ])

test_transforms = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ])

dataset_train = datasets.CIFAR10(root='./data', train=True, transform=train_transforms, download=True)
dataset_test = datasets.CIFAR10(root='./data', train=False, transform=test_transforms, download=True)

train_iterator = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
test_iterator = DataLoader(dataset_test, batch_size=batch_size)

print(f'Number of training examples: {len(dataset_train)}')
print(f'Number of testing examples: {len(dataset_test)}')
# Print in color for test/train logs
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def SelectiveFedAvg(w_locals, selected_layers):
    """
    Perform federated averaging on specific layers.
    
    Args:
        w_locals: List of state_dicts from all clients.
        selected_layers: List of layer names to selectively aggregate.
        
    Returns:
        w_glob: Global model with selectively averaged layers.
    """
    w_glob = copy.deepcopy(w_locals[0])  # Start with the first client's model

    for layer_name in w_glob.keys():
        if layer_name in selected_layers:
            # Aggregate this layer across all clients
            for i in range(1, len(w_locals)):
                w_glob[layer_name] += w_locals[i][layer_name]
            w_glob[layer_name] = torch.div(w_glob[layer_name], len(w_locals))
        else:
            # Keep other layers as they are (e.g., from the first client)
            pass

    return w_glob
    
def load_client_model(idx, base_model):
    model = copy.deepcopy(base_model)
    path = client_ckpt_path(idx)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded client {idx} model from disk")

    return model
    
def save_and_unload_client(model, idx):
    torch.save(model.state_dict(), client_ckpt_path(idx))
    #print(f"2. GPU memory before deleting model: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    del model
    torch.cuda.empty_cache()
    #print(f"2. GPU memory after deleting model and emptying cache: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

#Compute the outputdimensions for the client-side and the server-side models
def compute_output_dim(module, input_size):
    def compute_window_output(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) / stride) + 1
    
    if isinstance(module, nn.Conv2d):
        output_size = compute_window_output(input_size=input_size, kernel_size=module.kernel_size[0], padding=module.padding[0], stride=module.stride[0])
    elif isinstance(module, nn.MaxPool2d):
        output_size = compute_window_output(input_size=input_size, kernel_size=module.kernel_size, padding=module.padding, stride=module.stride)
    else:
        return input_size
    return output_size

# Dynamically splits the AlexNet model into client-side and server-side models
def split_model_at_layer(cut_layer, input_size, input_channels, num_classes):
    layer_name_to_idx = {"conv1": 3, "conv2": 6, "conv3": 8, "conv4": 11, "conv5": 13, "conv6": 16, "conv7": 18, "conv8": 21}
    cut_layer_idx = layer_name_to_idx[cut_layer]
    output_size = input_size

    class OurVGG11(nn.Module):
      def __init__(self, num_classes, input_channels):
        super(OurVGG11, self).__init__()
        self.features = nn.Sequential(
          nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
          nn.Linear(in_features=25088, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )
      def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x

    client_side_model = nn.Sequential()
    server_side_model = nn.Sequential()
    set_seeds(seed=1234)
    vgg11 = OurVGG11(num_classes, input_channels)

    for name, module in vgg11.named_children():
        if name != "features":
          continue
        for sub_name, sub_module in module.named_children():
            if int(sub_name) < cut_layer_idx:
                client_side_model.add_module(sub_name, sub_module)
            else:
                server_side_model.add_module(sub_name, sub_module)
            output_size = int(compute_output_dim(sub_module, input_size=output_size))

    x = torch.randn(1, input_channels, input_size, input_size)
    client_output = client_side_model(x)
    server_out = server_side_model(client_output)

    print(client_output.shape, server_out.shape)
    print(server_out.shape[1] * output_size * output_size)

    set_seeds(seed=1234)
    server_model_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d(output_size=(7, 7)),
          nn.Flatten(),
          nn.Linear(in_features=25088, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=num_classes, bias=True)
          )

    full_server_model = nn.Sequential(*server_side_model, *server_model_fc)

    return client_side_model, full_server_model, output_size, client_output.shape


# Calculate accuracy
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc
# Function to get the size of the output from the client-side model

#Initialize parameters for the dynamic split
#Cut layers can be one of the following: conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8 
input_size = 224
client_side_model, full_server_model, output_size, client_output_shape = split_model_at_layer(cut_layer, input_size, input_channels=3, num_classes=10)     
print("Client Model:", client_side_model)
print("Full Server Model:", full_server_model)

#net_glob_client = AlexNet_client_side()
net_glob_client = client_side_model
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    #net_glob_client = nn.DistributedDataParallel(net_glob_client) 
    net_glob_client = nn.DataParallel(net_glob_client)    

net_glob_client.to(device)
print("client-side model's architecture")
print(net_glob_client)     
feature_size = client_output_shape #may not be used

net_glob_server=full_server_model
print("Feature size--")
print(feature_size)
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    #net_glob_server = nn.DistributedDataParallel(net_glob_server) 
    net_glob_server = nn.DataParallel(net_glob_server)   # To use multiple GPUs 

net_glob_server.to(device)
print("server-side model's architecture")
print(net_glob_server)      
#==============================================================================================================
#                                       Federated Learning Program Functions
#==============================================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [None for i in range(num_users)]  # placeholders, load from disk when needed
net_server = None  # will be loaded per client

#Server-side Training
# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr

    #print(f"7. GPU memory before loading net_server to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    # Load server model for this client
    net_server = copy.deepcopy(net_glob_server)
    try:
        if os.path.exists(server_ckpt_path(idx)):
            net_server.load_state_dict(torch.load(server_ckpt_path(idx), map_location="cpu", weights_only=True))
    except Exception as e:
        print(f"Warning: Failed to load server checkpoint for client {idx}: {e}. Using global model.")
    net_server = net_server.to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    optimizer_server.zero_grad()
    fx_client = fx_client.to(device)
    y = y.to(device)
    # Forward pass on the server model
    fx_server = net_server(fx_client)

    # Calculate server-side loss and accuracy
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    # Backward pass
    loss.backward()
    optimizer_server.step()

    # Log metrics
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Save server model to disk
    torch.save(net_server.state_dict(), server_ckpt_path(idx))

    count1 += 1

    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_loss_train.clear()
        batch_acc_train.clear()
        count1 = 0

        prRed(f'Client{idx} Train => Local Epoch: {l_epoch_count} \tAcc: {acc_avg_train:.3f} \tLoss: {loss_avg_train:.4f}')
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            #print("nai---l_epcoch_count---")
            print(l_epoch_count)
            print(l_epoch)
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 

            ##TODO: Correct the w_locals_server appending logic.
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
        
        del net_server
        torch.cuda.empty_cache()
           
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True
            
            # Load all server state_dicts incrementally to avoid OOM
            if num_users > 0:
                w_glob_server = torch.load(server_ckpt_path(0), map_location="cpu")
                for i in range(1, num_users):
                    w = torch.load(server_ckpt_path(i), map_location="cpu")
                    for k in w_glob_server.keys():
                        w_glob_server[k] += w[k]
                    del w
                    torch.cuda.empty_cache()
                for k in w_glob_server.keys():
                    w_glob_server[k] = torch.div(w_glob_server[k], num_users)   
            
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)    
            # Save global to all server checkpoints
            for i in range(num_users):
                torch.save(w_glob_server, server_ckpt_path(i))
            
            w_locals_server = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []            
    # send gradients to the client               
    #return dfx_client
# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    #net = copy.deepcopy(net_model_server[idx]).to(device)
    net = copy.deepcopy(net_glob_server)
    net.load_state_dict(torch.load(server_ckpt_path(idx), map_location="cpu"))
    net = net.to(device)
    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))

            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

                del loss
                torch.cuda.empty_cache()
                del net
                torch.cuda.empty_cache()
                del fx_server    
                torch.cuda.empty_cache()
                del fx_client    
                torch.cuda.empty_cache()
                del y
                torch.cuda.empty_cache()    
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------ Federation process at Server-Side ------- ")
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
            
         
    return 
#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, fc_dim=client_output_shape[1]*client_output_shape[2]*client_output_shape[3]):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1  # Number of local epochs
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=batch_size, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=batch_size, shuffle=True)

        # Temporary fully connected layer at the client's final output
        set_seeds(seed=1234)
        self.fc_local=nn.Linear(fc_dim, 10) #10 is the number of classes for CIFAR10.
        self.fc_local.to(device)
        self.criterion_local = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        self.fc_local.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        optimizer_fc_local = torch.optim.Adam(self.fc_local.parameters(), lr=self.lr)
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #print(f"6. GPU memory before moving batch to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                images, labels = images.to(self.device), labels.to(self.device)
                #print(f"6. GPU memory after moving batch to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                optimizer_client.zero_grad()
                optimizer_fc_local.zero_grad()
               # Forward pass through client model
                features = net(images)
                client_fx = features.clone().detach().requires_grad_(True)
                # Sending activations to server and receiving gradients from server
                train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                del client_fx
                torch.cuda.empty_cache()

                #TODO: We want here to insert adaptively the cut layer's parameters.
                features = features.view(features.size(0), -1) 
                # Compute local loss using the temporary FC layer
                outputs = self.fc_local(features)
                loss_local = self.criterion_local(outputs, labels)

                # Backward pass on client model and local FC layer
                loss_local.backward()
                del features
                torch.cuda.empty_cache()
                optimizer_client.step()
                optimizer_fc_local.step()

                del images, labels
                torch.cuda.empty_cache()
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 

    def evaluate(self, net, ell):
        net.eval()
        self.fc_local.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                #print(f"12. GPU memory before moving test batch to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward pass
                features = net(images)
                #outputs = self.fc_local(features)
                # Send activations to the server for server-side evaluation
                evaluate_server(features, labels, self.idx, len_batch, ell)

def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
#----------------------------------------------------------------
def dataset_non_iid(dataset, num_users, num_shards=5000, num_samples_per_shard=12):
    """
    Splits dataset among num_users in a non-IID manner using shard-based partitioning
    as described in McMahan et al., 2017.

    Args:
        dataset (torch dataset): The dataset to split.
        num_users (int): Number of users.
        num_shards (int): Total number of shards (default: 5000 for MNIST).
        num_samples_per_shard (int): Number of samples per shard (default: 12 for MNIST).

    Returns:
        dict: Dictionary where keys are user IDs and values are sets of dataset indices.
    """
    num_items = len(dataset)
    #assert num_items == num_shards * num_samples_per_shard, "Dataset size must match total shard size."

    dict_users = {i: np.array([]) for i in range(num_users)}
    all_idxs = np.arange(num_items)
    labels = np.array(dataset.targets)  # MNIST/FashionMNIST labels

    # Sort the data by label
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # Sort by label
    sorted_idxs = idxs_labels[0, :]  # Sorted indices

    # Split data into shards
    shards = np.array_split(sorted_idxs, num_shards)
    shard_idxs = np.arange(num_shards)
    np.random.shuffle(shard_idxs)  # Shuffle shards to randomize allocation

    # Assign shards to users
    num_shards_per_user = num_shards // num_users
    for i in range(num_users):
        chosen_shards = shard_idxs[i * num_shards_per_user: (i + 1) * num_shards_per_user]
        for shard in chosen_shards:
            dict_users[i] = np.concatenate((dict_users[i], shards[shard]), axis=0)

    # Convert indices to integers
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int)  # Convert to int

    return dict_users

dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()

# Grouping clients into two groups
group1 = range(0,19)
group2 = range(20,29)

# Initialize separate global models for each group
w_glob_client1 = copy.deepcopy(w_glob_client)
w_glob_client2 = copy.deepcopy(w_glob_client)

# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

# CONV 8 AGGR LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias", "8.weight", "8.bias", "11.weight", "11.bias", "13.weight", "13.bias", "16.weight", "16.bias", "18.weight", "18.bias"]

# CONV 7 AGGR LAYERS
# selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias", "8.weight", "8.bias", "11.weight", "11.bias", "13.weight", "13.bias", "16.weight", "16.bias"]

# CONV 6 AGGR LAYERS
#selected_layers = ["6.weight", "6.bias", "8.weight", "8.bias", "11.weight", "11.bias", "13.weight", "13.bias"]

# CONV 5 AGGR LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias", "8.weight", "8.bias", "11.weight", "11.bias"]

# CONV 4 AGGR LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias","8.weight", "8.bias"]

# CONV 3 AGGR LAYERS
# selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias"]

# CONV 2 AGGR LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias"]

def get_selected_layers(collaborative_layer, cut_layer):
    """
    Returns the selected layers for selective federated averaging from collaborative_layer to cut_layer inclusive.
    """
    conv_stages = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8"]
    layer_dict = {
        "conv1": ["0.weight", "0.bias"],
        "conv2": ["3.weight", "3.bias"],
        "conv3": ["6.weight", "6.bias"],
        "conv4": ["8.weight", "8.bias"],
        "conv5": ["11.weight", "11.bias"],
        "conv6": ["13.weight", "13.bias"],
        "conv7": ["16.weight", "16.bias"],  # Assuming extended VGG
        "conv8": ["18.weight", "18.bias"],  # Assuming extended VGG
    }
    start_idx = conv_stages.index(collaborative_layer)
    end_idx = conv_stages.index(cut_layer)
    selected = []
    for i in range(start_idx, end_idx + 1):
        selected.extend(layer_dict[conv_stages[i]])
    return selected

selected_layers = get_selected_layers(collaborative_layer, cut_layer)


for iter in range(global_epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)

    # Separate lists to collect local weights for each group
    w_locals_client1 = []
    w_locals_client2 = []

    for idx in idxs_users:
        set_seeds(seed=1234)
        # Training ------------------
        #w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        net_client = load_client_model(idx, net_glob_client)
        #print(f"1. GPU memory before moving net_client to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        net_client = net_client.to(device)
        #print(f"1. GPU memory after moving net_client to GPU: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx],     fc_dim=client_output_shape[1]*client_output_shape[2]*client_output_shape[3])        
        w_client = local.train(net_client)
        net_client.to('cpu')
        save_and_unload_client(net_client, idx)
        
        # Add the local model weights to the appropriate group
        if idx in group1:
            w_locals_client1.append(w_client)
        elif idx in group2:
            w_locals_client2.append(w_client)
        
        # Testing -------------------
        net_eval = load_client_model(idx, net_glob_client).to(device)
        local.evaluate(net_eval, ell=iter)
       
        del w_client
        torch.cuda.empty_cache()
        del net_eval
        torch.cuda.empty_cache()
        #print(f"4. GPU memory after deleting net_eval and emptying cache: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        del local
        torch.cuda.empty_cache()
    # Round process completed    
    # Save to Excel after each round
    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect})
    file_name = program + ".xlsx"
    df.to_excel(file_name, sheet_name="v1_test", index=False)           
    # Ater serving all clients for its local epochs------------
   
    # Selective federated averaging for group 1 (incremental to save memory)
    if w_locals_client1:
        w_glob_client1 = copy.deepcopy(w_locals_client1[0])
        for client_w in w_locals_client1[1:]:
            for k in w_glob_client1.keys():
                if k in selected_layers:
                    w_glob_client1[k] += client_w[k]
                # Non-selected layers keep from first client
        # Divide selected layers by number of clients
        num_clients1 = len(w_locals_client1)
        for k in selected_layers:
            if k in w_glob_client1:
                w_glob_client1[k] = torch.div(w_glob_client1[k], num_clients1)

    # Selective federated averaging for group 2 (incremental to save memory)
    if w_locals_client2:
        w_glob_client2 = copy.deepcopy(w_locals_client2[0])
        for client_w in w_locals_client2[1:]:
            for k in w_glob_client2.keys():
                if k in selected_layers:
                    w_glob_client2[k] += client_w[k]
                # Non-selected layers keep from first client
        # Divide selected layers by number of clients
        num_clients2 = len(w_locals_client2)
        for k in selected_layers:
            if k in w_glob_client2:
                w_glob_client2[k] = torch.div(w_glob_client2[k], num_clients2)
    
    # Clear the lists to free memory
    del w_locals_client1
    del w_locals_client2
    # Fed  Server: Federation process at Client-Side-----------
    if (iter + 1) % fedavg_freq == 0: #update and the aggregators' models and the whole models
        print("Federated Averaging")
        # Combine group aggregations to update global model
        if w_glob_client1 is not None and w_glob_client2 is not None:
            w_glob_combined = copy.deepcopy(w_glob_client1)
            for k in w_glob_combined.keys():
                w_glob_combined[k] += w_glob_client2[k]
                w_glob_combined[k] = torch.div(w_glob_combined[k], 2)  # Average of 2 groups

        if 'w_glob_combined' in locals():
            net_glob_client.load_state_dict(w_glob_combined)
    
    else: #update only the aggregators' models!
        # Assign global weights back to clients
        for idx in idxs_users:
            #w_locals_client.append(copy.deepcopy(w_client))
            if idx in group1:
                net_glob_client.load_state_dict(w_glob_client1)
            elif idx in group2:
                net_glob_client.load_state_dict(w_glob_client2)
#===================================================================================     
print("Training and Evaluation completed!")    
end = time.time()
print(end - start)
# Function to get the size of the output from the client-side model
def get_feature_size(model, input_size=(3, 64, 64)):
    with torch.no_grad():
        x = torch.randn(1, *input_size)
        x = model(x)
        return x.view(1, -1).size(1)