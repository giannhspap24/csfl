import matplotlib.pyplot as plt
import pandas as pd



# Load the datasets
df_csfl = pd.read_excel("alexnet_MNIST_conv5_100clients_10_agg_iid_300_epochs.xlsx")
df_sfl_tirana = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_sfl = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_accsfl = pd.read_excel("alexnet_MNIST_conv5_100clients_100_agg_iid_300_epochs.xlsx")

# Extract accuracies and epochs
full_accuracies1, epochs1 = df_csfl['acc_test'], df_csfl['round']
full_accuracies2, epochs2 = df_sfl_tirana['acc_test'], df_sfl_tirana['round']
full_accuracies3, epochs3 = df_sfl['acc_test'], df_sfl['round']
full_accuracies4, epochs4 = df_accsfl['acc_test'], df_accsfl['round']

layer_mbytes_parameters=[1,7,8,7,10,80,30,10]#for MNIST dataset
layer_mbytes_activations=[0.01,0.3,1.1,2.2,4.5,80,30,10]#for MNIST dataset

av=0.3 #cut layer parameters (in MB)
ah=0.07 #collaborative layer parameters (in MB)
B=10 #number of batches 
N=100 #number of clients
lamda=0.25 #percentage of aggregators
v=4
h=1

comm_per_round_sfl=((2*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_sfl_tirana=((2 *2*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_accsfl=((1*layer_mbytes_activations[v]*B + 2 * sum(layer_mbytes_parameters[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_csfl=(((layer_mbytes_activations[h]*B*2 + 2* sum(layer_mbytes_parameters[:h])) * (lamda*N)) + (layer_mbytes_activations[v]*B *N) + (2* sum(layer_mbytes_parameters[h:v])) ) / 1000 #convert MB to GB
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


# Convert GB to TB
comm_csfl /= 1000
comm_sfl /= 1000
comm_accsfl /= 1000
comm_sfl_tirana /= 1000
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
plt.plot(comm_csfl, full_accuracies_smoothed1,linestyle='--', color='c', label='SFL3', linewidth=4,markersize=1)
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
plt.savefig("accuracy_vs_overhead_mnist.pdf", format="pdf", bbox_inches="tight")

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