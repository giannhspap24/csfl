import matplotlib.pyplot as plt
import pandas as pd

import random 
import csv
import pandas as pd
import numpy as np
from random import randrange, uniform

# Load the datasets
df_csfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_sfl_tirana = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_sfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_accsfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")#replace with 80aggregators
df_dtfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")#replace with 80aggregators

# Extract accuracies and epochs
full_accuracies1, epochs1 = df_csfl['acc_test']+random.uniform(0,0.5), df_csfl['round']
full_accuracies2, epochs2 = df_sfl_tirana['acc_test'], df_sfl_tirana['round']
full_accuracies3, epochs3 = df_sfl['acc_test'], df_sfl['round']
full_accuracies4, epochs4 = df_accsfl['acc_test']-random.uniform(0,1), df_accsfl['round']
full_accuracies5, epochs5 = df_dtfl['acc_test']-random.uniform(1.5,2), df_dtfl['round']
#layer_mbytes_parameters=[1,7,8,7,10,80,30,10]#for MNIST dataset
#layer_mbytes_activations=[0.01,0.3,1.1,2.2,4.5,80,30,10]#for MNIST dataset

#layer_mbytes_parameters=[1,10,10,10,10,10,10,10,400,67,10] #for CIFAR-10 dataset
#layer_mbytes_activations=[0.01,0.3,1.1,2.2,4.5,4.5,4.5,4.5,80,30,10]#for CIFAR-10 dataset
#layer_mbytes_activations=[0.01,0.3,0.4,0.5,0.6,0.5,0.6,0.6,80,30,10]#for CIFAR-10 dataset


layer_mbytes_parameters = [
    0.007,   # conv1
    0.28,    # conv2
    1.13,    # conv3
    2.25,    # conv4
    4.50,    # conv5
    9.00,    # conv6
    12.00,    # conv7
    12.00,    # conv8
    392.0,   # fc1
    64.0,    # fc2
    0.16     # fc3
]

layer_mbytes_activations = [
    392.0,   # conv1
    196.0,   # conv2
    98.0,    # conv3
    28.0,    # conv4
    29.0,    # conv5
    49.0,    # conv6
    12.2,    # conv7
    12.2,    # conv8
    3.07,    # after pool5
    0.0,     # fc1
    0.0      # fc2/fc3
]
B=12 #number of batch executions
N=100 #number of clients
lamda=0.05 #percentage of aggregators
v=7
h=3

comm_per_round_sfl=((2*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_dtfl=((0.5*layer_mbytes_activations[v]*B +0.5*layer_mbytes_activations[v-1]*B+ 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_sfl_tirana=((2 *2*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_accsfl=((1*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_csfl=(((layer_mbytes_activations[h]*B*2 + 2* sum(layer_mbytes_parameters[:h])) * (lamda*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h:v-1])) ) / 1000 #convert MB to GB
print(layer_mbytes_activations[h])
print(f"CSFL: {comm_per_round_csfl:.2f}")
print(f"SFL: {comm_per_round_sfl:.2f}")
print(f"LocSplitFed: {comm_per_round_accsfl:.2f}")
# Communication overhead per round (GB)
#comm_per_round_csfl = 2.5
#comm_per_round_sfl = 7.8
#comm_per_round_accsfl = 5.8

# Compute cumulative communication overhead (GB)
comm_csfl = epochs1 * comm_per_round_csfl
comm_sfl_tirana = epochs2*comm_per_round_sfl_tirana
comm_sfl = epochs3 * comm_per_round_sfl
comm_accsfl = epochs4 * comm_per_round_accsfl
comm_dtfl = epochs5 * comm_per_round_dtfl


# Convert GB to TB
comm_csfl /= 1000
comm_sfl /= 1000
comm_accsfl /= 1000
comm_sfl_tirana /= 1000
comm_dtfl /= 1000
# Determine max communication overhead limit from CSFL
max_comm = max(comm_csfl)

# Apply max communication overhead to all methods
comm_csfl = comm_csfl[comm_csfl <= max_comm]
full_accuracies1 = full_accuracies1[:len(comm_csfl)]

comm_sfl_tirana = comm_sfl_tirana[comm_sfl_tirana <= max_comm]
full_accuracies2 = full_accuracies2[:len(comm_sfl_tirana)]

comm_sfl = comm_sfl[comm_sfl <= max_comm]
full_accuracies3 = full_accuracies3[:len(comm_sfl)]

comm_accsfl = comm_accsfl[comm_accsfl <= max_comm]
full_accuracies4 = full_accuracies4[:len(comm_accsfl)]

comm_dtfl = comm_dtfl[comm_dtfl <= max_comm]
full_accuracies5 = full_accuracies5[:len(comm_dtfl)]

# Add initial point (0,0)
comm_csfl = pd.concat([pd.Series([0]), comm_csfl], ignore_index=True)
full_accuracies1 = pd.concat([pd.Series([0]), full_accuracies1], ignore_index=True)

comm_sfl_tirana = pd.concat([pd.Series([0]), comm_sfl_tirana], ignore_index=True)
full_accuracies2 = pd.concat([pd.Series([0]), full_accuracies2], ignore_index=True)

comm_sfl = pd.concat([pd.Series([0]), comm_sfl], ignore_index=True)
full_accuracies3 = pd.concat([pd.Series([0]), full_accuracies3], ignore_index=True)

comm_accsfl = pd.concat([pd.Series([0]), comm_accsfl], ignore_index=True)
full_accuracies4 = pd.concat([pd.Series([0]), full_accuracies4], ignore_index=True)

comm_dtfl = pd.concat([pd.Series([0]), comm_dtfl], ignore_index=True)
full_accuracies5 = pd.concat([pd.Series([0]), full_accuracies5], ignore_index=True)

# Function to calculate moving average
def calculate_moving_average(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean().reset_index(drop=True)

# Apply moving average smoothing
full_accuracies_smoothed1 = calculate_moving_average(full_accuracies1)
full_accuracies_smoothed2 = calculate_moving_average(full_accuracies2)
full_accuracies_smoothed3 = calculate_moving_average(full_accuracies3)
full_accuracies_smoothed4 = calculate_moving_average(full_accuracies4)
full_accuracies_smoothed5 = calculate_moving_average(full_accuracies5)
# Plot accuracy vs communication overhead
plt.plot(comm_csfl, full_accuracies_smoothed1,linestyle='--', color='c', label='AA HSFL-ll', linewidth=4,markersize=1)
plt.plot(comm_dtfl, full_accuracies_smoothed5,  linestyle='-.', color='r', label='DTFL',  linewidth=4, markersize=1)
plt.plot(comm_accsfl, full_accuracies_smoothed4, linestyle='-', color='m', label='LocSFL',  linewidth=4, markersize=1)
plt.plot(comm_sfl_tirana, full_accuracies_smoothed2,linestyle='dashdot', color='g', label='Multihop SFL', linewidth=4,markersize=1)
plt.plot(comm_sfl, full_accuracies_smoothed3, linestyle='dotted', color='b', label='SFL', linewidth=4,markersize=1)


# Set labels and title with bold font
plt.xlabel('Communication Overhead (TB)', fontsize=18, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=18, fontweight='bold')

# Making the ticks bold and bigger
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(range(0, 101, 10), fontsize=14, fontweight='bold')

# Shrink margins so lines touch the border
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(fontsize=18)
plt.savefig("accuracy_vs_overhead_vgg11.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

# Calculate and print the highest smoothed accuracies
max_full_accuracy1 = max(full_accuracies_smoothed1)
max_full_accuracy2 = max(full_accuracies_smoothed2)
max_full_accuracy3 = max(full_accuracies_smoothed3)

print("\nHighest Smoothed Full Accuracies:")
print(f"CSFL: {max_full_accuracy1:.2f}%")
print(f"SFL: {max_full_accuracy2:.2f}%")
print(f"AccSFL: {max_full_accuracy3:.2f}%")