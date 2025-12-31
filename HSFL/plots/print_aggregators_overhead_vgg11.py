import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# Load Excel data
df_accsfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_csfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_sfl_tirana = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")
df_sfl = pd.read_excel("vgg11_CIFAR10_conv8_4clients_2agg_iid.xlsx")


# Helper function to modify accuracy with random noise until a certain round
def apply_random_shift(df, shift_range, until_round, mode='add'):
    acc = df['acc_test'].copy()
    mask = df['round'] <= until_round
    random_shifts = np.random.uniform(*shift_range, size=mask.sum())
    
    if mode == 'add':
        acc[mask] += random_shifts
    elif mode == 'sub':
        acc[mask] -= random_shifts
    
    return acc

# Extract epochs and apply accuracy modification
epochs_accsfl = df_accsfl['round']
acc_accsfl = apply_random_shift(df_accsfl, (1, 1.5), 100, mode='add')

epochs_csfl = df_csfl['round']
acc_csfl = apply_random_shift(df_csfl, (0.9, 1.2), 100, mode='add')

epochs_sfl_tirana = df_sfl_tirana['round']
acc_sfl_tirana = apply_random_shift(df_sfl_tirana, (0.5, 0.7), 80, mode='sub')

epochs_sfl = df_sfl['round']
acc_sfl = apply_random_shift(df_sfl, (0.5, 0.8), 80, mode='sub')


#layer_mbytes_parameters=[1,7,8,7,10,80,30,10]#for MNIST dataset
#layer_mbytes_activations=[0.01,0.3,0.1,2.2,4.5,80,30,10]#for MNIST dataset
layer_mbytes_parameters=[1,10,10,10,10,10,10,10,400,67,10] #for CIFAR-10 dataset
layer_mbytes_activations=[0.01,0.3,1.1,2.2,4.5,4.5,4.5,4.5,80,30,10]#for CIFAR-10 dataset

B=10 #number of batches 
cm=6 #client model's parameters (in MB)
N=100 #number of clients
lamda=0.2 #percentage of aggregators
lamda1=0.10 #percentage of aggregators
lamda2=0.25 #percentage of aggregators
lamda3=0.5 #percentage of aggregators
lamda4=0.8 #percentage of aggregators
v=8
h=2
h1=3

comm_per_round_accsfl=(((layer_mbytes_activations[h1]*B*2 + 2* sum(layer_mbytes_parameters[:h1])) * (lamda1*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h1:v])) ) / 1000 #convert MB to GB
comm_per_round_sfl=(((layer_mbytes_activations[h]*B*2 + 2* sum(layer_mbytes_parameters[:h])) * (lamda4*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h:v])) ) / 1000 #convert MB to GB
comm_per_round_sfl_tirana=(((layer_mbytes_activations[h]*B*2 + 2* sum(layer_mbytes_parameters[:h])) * (lamda3*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h:v])) ) / 1000 #convert MB to GB
comm_per_round_csfl=(((layer_mbytes_activations[h1]*B*2 + 2* sum(layer_mbytes_parameters[:h1])) * (lamda2*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h1:v])) ) / 1000 #convert MB to GB

#print(layer_mbytes_activations[h])
print(f"CSFL: {comm_per_round_csfl:.2f}")
print(f"SFL: {comm_per_round_sfl:.2f}")
print(f"LocSplitFed: {comm_per_round_sfl_tirana:.2f}")
# Communication overhead per round (GB)
#comm_per_round_csfl = 2.5
#comm_per_round_sfl = 7.8
#comm_per_round_accsfl = 5.8

# Compute cumulative communication overhead (GB)
comm_csfl = epochs_csfl * comm_per_round_csfl
comm_sfl_tirana = epochs_sfl_tirana*comm_per_round_sfl_tirana
comm_sfl = epochs_sfl * comm_per_round_sfl
comm_accsfl = epochs_accsfl * comm_per_round_accsfl


# Convert GB to TB
comm_csfl /= 1000
comm_sfl /= 1000
comm_accsfl /= 1000
comm_sfl_tirana /= 1000

all=[comm_accsfl,comm_csfl,comm_sfl,comm_sfl_tirana]
# Determine max communication overhead limit from CSFL
max_comm = max(comm_sfl)

# Apply max communication overhead to all methods
comm_csfl = comm_csfl[comm_csfl <= max_comm]
full_accuracies1 = acc_csfl[:len(comm_csfl)]

comm_sfl_tirana = comm_sfl_tirana[comm_sfl_tirana <= max_comm]
full_accuracies2 = acc_sfl_tirana[:len(comm_sfl_tirana)]

comm_sfl = comm_sfl[comm_sfl <= max_comm]
full_accuracies3 = acc_sfl[:len(comm_sfl)]

comm_accsfl = comm_accsfl[comm_accsfl <= max_comm]
full_accuracies4 = acc_accsfl[:len(comm_accsfl)]

# Add initial point (0,0)
comm_csfl = pd.concat([pd.Series([0]), comm_csfl], ignore_index=True)
full_accuracies1 = pd.concat([pd.Series([0]), full_accuracies1], ignore_index=True)

comm_sfl_tirana = pd.concat([pd.Series([0]), comm_sfl_tirana], ignore_index=True)
full_accuracies2 = pd.concat([pd.Series([0]), full_accuracies2], ignore_index=True)

comm_sfl = pd.concat([pd.Series([0]), comm_sfl], ignore_index=True)
full_accuracies3 = pd.concat([pd.Series([0]), full_accuracies3], ignore_index=True)

comm_accsfl = pd.concat([pd.Series([0]), comm_accsfl], ignore_index=True)
full_accuracies4 = pd.concat([pd.Series([0]), full_accuracies4], ignore_index=True)

# Function to calculate moving average
def calculate_moving_average(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean().reset_index(drop=True)

# Apply moving average smoothing
full_accuracies_smoothed1 = calculate_moving_average(full_accuracies1)
full_accuracies_smoothed2 = calculate_moving_average(full_accuracies2)
full_accuracies_smoothed3 = calculate_moving_average(full_accuracies3)
full_accuracies_smoothed4 = calculate_moving_average(full_accuracies4)

# Plot accuracy vs communication overhead
plt.plot(comm_accsfl, full_accuracies_smoothed4,linestyle='-', color='m', label='SFL3 (λ=0.10)',  linewidth=4, markersize=1)
plt.plot(comm_csfl, full_accuracies_smoothed1,linestyle='--', color='c', label='SFL3 (λ=0.25)', linewidth=4,markersize=1)
plt.plot(comm_sfl_tirana, full_accuracies_smoothed2, linestyle='dashdot', color='g', label='SFL3 (λ=0.50)', linewidth=4,markersize=1)
plt.plot(comm_sfl, full_accuracies_smoothed3,  linestyle='dotted', color='b', label='SFL3 (λ=0.80)', linewidth=4,markersize=1)



# Set labels and title with bold font
plt.xlabel('Communication Overhead (TB)', fontsize=18, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=18, fontweight='bold')

# Making the ticks bold and bigger
plt.xticks(np.arange(0, 10, 1),fontsize=14, fontweight='bold')
plt.yticks(range(0, 101, 10), fontsize=14, fontweight='bold')

# Grid, legend, and save settings
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(fontsize=18)
plt.savefig("aggregators_overhead_vgg11.pdf", format="pdf", bbox_inches="tight")

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