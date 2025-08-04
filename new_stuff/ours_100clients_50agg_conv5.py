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
num_users = 100
global_epochs = 330
frac = 1        
lr = 0.0001
#Program identifier
program = "alexnet_MNIST_conv5_100clients_10_agg_iid_300_epochs"
cut_layer = "conv5"  
print(f"---------{program}----------")

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#.to(torch.device("cpu"))
# MNIST dataset download and preparation
train_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
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
def split_model_at_layer(cut_layer, input_size, conv_layers, conv_kernel_sizes, conv_kernel_strides, conv_kernel_paddings, pool_kernel_sizes, pool_kernel_strides):
    layer_name_to_idx = {"conv1": 3, "conv2": 6, "conv3": 9, "conv4": 12, "conv5": 14, "fc1": 14, "fc2": 14}
    cut_layer_idx = layer_name_to_idx[cut_layer]
    output_size = input_size

    class OurAlexNetClient(nn.Module):
        def __init__(self, num_classes=10):
            super(OurAlexNetClient, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(conv_layers[0], conv_layers[1], kernel_size=conv_kernel_sizes[0], stride=conv_kernel_strides[0], padding=conv_kernel_paddings[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[0], stride=pool_kernel_strides[0]),
                nn.Conv2d(conv_layers[1], conv_layers[2], kernel_size=conv_kernel_sizes[1], stride=conv_kernel_strides[1], padding=conv_kernel_paddings[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[1], stride=pool_kernel_strides[1]),
                nn.Conv2d(conv_layers[2], conv_layers[3], kernel_size=conv_kernel_sizes[2], stride=conv_kernel_strides[2], padding=conv_kernel_paddings[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[2], stride=pool_kernel_strides[2]),
                nn.Conv2d(conv_layers[3], conv_layers[4], kernel_size=conv_kernel_sizes[3], stride=conv_kernel_strides[3], padding=conv_kernel_paddings[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[3], stride=pool_kernel_strides[3]),
                nn.Conv2d(conv_layers[4], conv_layers[5], kernel_size=conv_kernel_sizes[4], stride=conv_kernel_strides[4], padding=conv_kernel_paddings[4]),
                nn.ReLU(),
            )

        def forward(self, x):
            x = self.model(x)
            return x

    client_side_model = nn.Sequential()
    server_side_model = nn.Sequential()
    set_seeds(seed=1234)
    alexnet = OurAlexNetClient()

    for name, module in alexnet.named_children():
        for sub_name, sub_module in module.named_children():
            if int(sub_name) < cut_layer_idx:
                client_side_model.add_module(sub_name, sub_module)
            else:
                server_side_model.add_module(sub_name, sub_module)
            output_size = int(compute_output_dim(sub_module, input_size=output_size))

    # Compute model output shapes for Linear Layer construction
    x = torch.randn(1, conv_layers[0], input_size, input_size)
    client_output = client_side_model(x)
    server_out = server_side_model(client_output)

    print(client_output.shape, server_out.shape)
    print(server_out.shape[1] * output_size * output_size)

    set_seeds(seed=1234)
    fc1 = nn.Linear(server_out.shape[1] * output_size * output_size, 512)
    set_seeds(seed=1234)
    fc2 = nn.Linear(512, 256)
    set_seeds(seed=1234)
    fc3 = nn.Linear(256, 10)

    server_model_fc = nn.Sequential(
        nn.Flatten(),
        fc1,
        nn.ReLU(),
        nn.Dropout(0.5),
        fc2,
        nn.ReLU(),
        nn.Dropout(0.5),
        fc3
    )

    if cut_layer == "fc1":
      client_model_fc = nn.Sequential(
        nn.Flatten(),
        fc1,
        nn.ReLU(),
      )

      server_model_fc = nn.Sequential(
        fc2,
        nn.ReLU(),
        fc3
      )
    
      full_server_model = nn.Sequential(*server_side_model, *server_model_fc)
      full_client_model = nn.Sequential(*client_side_model, *client_model_fc)

      return full_client_model, full_server_model, output_size, client_output.shape

    elif cut_layer == "fc2":
      client_model_fc = nn.Sequential(
        nn.Flatten(),
        fc1,
        nn.ReLU(),
        fc2,
        nn.ReLU()
      )

      server_model_fc = nn.Sequential(
        fc3
      )
    
      full_server_model = nn.Sequential(*server_side_model, *server_model_fc)
      full_client_model = nn.Sequential(*client_side_model, *client_model_fc)

      return full_client_model, full_server_model, output_size, client_output.shape

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
# cut layer can be one of the following: conv1, conv2, conv3, conv4, conv5, fc1, fc2
input_size = 28
conv_layers = [1, 32, 64, 128, 256, 512]
conv_kernel_sizes = [3, 3, 3, 3, 3]
conv_kernel_strides = [1, 1, 1, 1, 1]
conv_kernel_paddings = [1, 1, 1, 1, 1]
pool_kernel_sizes = [2, 2, 2, 2]
pool_kernel_strides = [2, 2, 2, 2]

client_side_model, full_server_model, output_size, client_output_shape = split_model_at_layer(
    cut_layer, input_size, conv_layers, conv_kernel_sizes, conv_kernel_strides, conv_kernel_paddings, pool_kernel_sizes, pool_kernel_strides
)     
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
#feature_size = get_feature_size(AlexNet_client_side())
#net_glob_server = AlexNet_server_side()
#feature_size = 512   # Adjust this to match the output size of the client-side network
feature_size = client_output_shape #may not be used

#net_glob_server = AlexNet_server_side(feature_size=feature_size)
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
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)
#                                       Server-side Training
# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    #global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train,w_locals_server
    #lobal loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train,loss_train_collect_user
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr

    net_server = copy.deepcopy(net_model_server[idx]).to(device)
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

    # Update the server model
    net_model_server[idx] = copy.deepcopy(net_server)

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
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                                   
            w_glob_server = FedAvg(w_locals_server)   
            
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)    
            net_model_server = [net_glob_server for i in range(num_users)]
            
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
    
    net = copy.deepcopy(net_model_server[idx]).to(device)
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
def evaluate_full_model_all_clients(net_glob_client, net_model_server, clients, device, round_num):
    """
    Evaluate full (client + server) model on each client's test data.

    Args:
        net_glob_client (nn.Module): The global client-side model.
        net_model_server (list): A list of each client's server-side model (after training).
        clients (list): A list of Client instances.
        device (torch.device): The device to run the models on.
        round_num (int): The current global round, used for saving progress.
        save_path (str): Path to save the Excel sheet.

    Returns:
        float: Average accuracy across all clients.
    """
    net_glob_client.eval()

    client_accuracies = []
    all_records = []

    with torch.no_grad():
        for idx, client in enumerate(clients):
            net_c = copy.deepcopy(net_glob_client).to(device)
            net_s = copy.deepcopy(net_model_server[idx]).to(device)
            net_c.eval()
            net_s.eval()

            correct = 0
            total = 0

            for images, labels in client.ldr_test:
                images, labels = images.to(device), labels.to(device)
                fx_client = net_c(images)
                outputs = net_s(fx_client)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            acc = 100.0 * correct / total if total > 0 else 0.0
            client_accuracies.append(acc)
            all_records.append({"Client": f"Client {idx}", "Accuracy": acc})
            print(f"[Round {round_num}] Client {idx} Accuracy: {acc:.2f}%")

    avg_acc = sum(client_accuracies) / len(client_accuracies)
    all_records.append({"Client": "Average", "Accuracy": avg_acc})
    print(f"\n[Round {round_num}] Average Accuracy: {avg_acc:.2f}%")

    # Save to Excel
    # --- Save results to Excel ---
    file_name = "full_model_accuracy_cut_layer_conv5_fedavgfrq_3_25_agg.xlsx"
   # round=round+1
    new_data = {
            "round": round_num,
            "acc_train": acc_avg_all_user_train,
            "acc_test": avg_acc
        }

    if os.path.exists(file_name):
            # Read existing data
        df = pd.read_excel(file_name)
            # Append new row
        df = df.append(new_data, ignore_index=True)
    else:
            # Create new dataframe
        df = pd.DataFrame([new_data])

        # Save dataframe back to excel
    df.to_excel(file_name, index=False)

    return avg_acc
# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, fc_dim=client_output_shape[1]*client_output_shape[2]*client_output_shape[3]):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1  # Number of local epochs
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=32, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=32, shuffle=True)

        # Temporary fully connected layer at the client's final output
        #TODO: Change the fully connected layer definition, after the split.
        #self.fc_local = nn.Linear(512, 10)  # 64: feature size, 10: number of classes
        set_seeds(seed=1234)
        self.fc_local=nn.Linear(fc_dim, 10)
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
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                optimizer_fc_local.zero_grad()
               # Forward pass through client model
                features = net(images)
                client_fx = features.clone().detach().requires_grad_(True)
                #print("Client model output shape:", features.shape)  # Expected: (batch_size, 64)
                #print(f"Feature size after convolution layers: {client_fx.numel()}")
                # Sending activations to server and receiving gradients from server
                train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                #print(dfx)
                #TODO: We want here to insert adaptively the cut layer's parameters.
                features = features.view(features.size(0), -1) 
                # Compute local loss using the temporary FC layer
                outputs = self.fc_local(features)
                loss_local = self.criterion_local(outputs, labels)

                # Backward pass on client model and local FC layer
                loss_local.backward()
                optimizer_client.step()
                optimizer_fc_local.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 

    def evaluate(self, net, ell):
        net.eval()
        self.fc_local.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
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
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()

num_groups = 10

# --- Grouping Clients and Initializing Models ---
clients_per_group = num_users // num_groups          # = 1
remainder = num_users % num_groups                   # = 20

groups = {}
start_idx = 0
for i in range(num_groups):
    end_idx = start_idx + clients_per_group + (1 if i < remainder else 0)
    groups[i] = list(range(start_idx, end_idx))
    start_idx = end_idx

w_glob_clients = {
    i: copy.deepcopy(w_glob_client)
    for i in range(num_groups)
}
# Federation takes place after certain local epochs in train() client-side

# this epoch is global epoch, also known as rounds
fedavg_freq=3

# CONV 5 SPLIT LAYERS
selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias","9.weight", "9.bias", "12.weight", "12.bias"]

# CONV4 SPLIT LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias", "9.weight", "9.bias"]

# CONV3 SPLIT LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias","6.weight", "6.bias"]

# CONV2 SPLIT LAYERS
#selected_layers = ["0.weight", "0.bias", "3.weight", "3.bias"]

for iter in range(global_epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)

    # --- Initialize local weights for each group ---
    w_locals_clients = {i: [] for i in range(num_groups)}

    # --- Train clients and collect their models ---
    for idx in idxs_users:
        set_seeds(seed=1234)
        local = Client(net_glob_client, idx, lr, device,
                       dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                       fc_dim=client_output_shape[1]*client_output_shape[2]*client_output_shape[3])
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))

        # Determine group of client and store weights
        for group_id, client_ids in groups.items():
            if idx in client_ids:
                w_locals_clients[group_id].append(copy.deepcopy(w_client))
                break

        # Optional: Evaluate client model
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

    # --- Group-level FedAvg (Selective) ---
    for group_id, w_locals in w_locals_clients.items():
        print(f"Group {group_id}: {len(w_locals)} clients")
        if not w_locals:
            continue

        base = copy.deepcopy(w_locals[0])
        for k in base.keys():
            if k in selected_layers:
                for i in range(1, len(w_locals)):
                    base[k] += w_locals[i][k]
                base[k] = torch.div(base[k], len(w_locals))
            else:
                base[k] = w_locals[0][k]  # Keep unchanged

        w_glob_clients[group_id] = base

    # --- Evaluate global model on all clients ---
    clients_eval = [
        Client(net_client_model=net_glob_client, idx=i, lr=lr, device=device,
               dataset_train=dataset_train, dataset_test=dataset_test,
               idxs=dict_users[i], idxs_test=dict_users_test[i])
        for i in range(num_users)
    ]
    evaluate_full_model_all_clients(net_glob_client=net_glob_client,
                                    net_model_server=net_model_server,
                                    clients=clients_eval,
                                    device=device,
                                    round_num=iter)

    # --- Federated Averaging across all group models ---
    if (iter + 1) % fedavg_freq == 0:
        print("Federated Averaging (Across Groups)")
        valid_group_ids = [gid for gid in w_locals_clients if w_locals_clients[gid]]
        if valid_group_ids:
            w_glob_combined = copy.deepcopy(w_glob_clients[valid_group_ids[0]])
            for k in w_glob_combined.keys():
                for gid in valid_group_ids[1:]:
                    w_glob_combined[k] += w_glob_clients[gid][k]
                w_glob_combined[k] = torch.div(w_glob_combined[k], len(valid_group_ids))

            net_glob_client.load_state_dict(w_glob_combined)

    else:
        # Local update only: assign back group models to each client
        for idx in idxs_users:
            for group_id, client_ids in groups.items():
                if idx in client_ids:
                    net_glob_client.load_state_dict(w_glob_clients[group_id])
                    break
    
#===================================================================================     
print("Training and Evaluation completed!")    
end = time.time()
print(end - start)
#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)

def get_feature_size(model, input_size=(3, 64, 64)):
    with torch.no_grad():
        x = torch.randn(1, *input_size)
        x = model(x)
        return x.view(1, -1).size(1)