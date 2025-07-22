import matplotlib.pyplot as plt
import pandas as pd



# Load the datasets
df_csfl = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")
df_sfl_tirana = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_sfl = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_accsfl = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")

# Extract accuracies and epochs
full_accuracies1, epochs1 = df_csfl['acc_test'], df_csfl['round']
full_accuracies2, epochs2 = df_sfl_tirana['acc_test'], df_sfl_tirana['round']
full_accuracies3, epochs3 = df_sfl['acc_test'], df_sfl['round']
full_accuracies4, epochs4 = df_accsfl['acc_test'], df_accsfl['round']

layer_mbytes=[1,7,8,7,10,80,30,10]#for MNIST dataset

av=0.3 #cut layer parameters (in MB)
ah=0.07 #collaborative layer parameters (in MB)
B=19 #number of batches 
cm=6 #client model's parameters (in MB)
am=5.91 #aggregators model's parameters (in MB)
wm=0.09 #weak model's parameters (in MB)
N=100 #number of clients
lamda=0.1 #percentage of aggregators
v=5
h=2

comm_per_round_sfl=((2*layer_mbytes[v]*B + 2 * sum(layer_mbytes[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_sfl_tirana=((2 *2*layer_mbytes[v]*B + 2 * sum(layer_mbytes[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_accsfl=((1*layer_mbytes[v]*B + 2 * sum(layer_mbytes[:v])) * N ) / 1000 #convert MB to GB
comm_per_round_csfl=(((layer_mbytes[h]*B*2 + 2* sum(layer_mbytes[:h])) * (lamda*N)) + (layer_mbytes[v]*B *N) + (2* sum(layer_mbytes[h:v])) ) / 1000 #convert MB to GB

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
plt.plot(comm_csfl, full_accuracies_smoothed1,linestyle='--', color='c', label='C-SFL', linewidth=4, markersize=1)
plt.plot(comm_sfl_tirana, full_accuracies_smoothed2,linestyle='-', color='g', label='SFL-helpers', linewidth=4, markersize=1)
plt.plot(comm_sfl, full_accuracies_smoothed3, linestyle='-', color='b', label='SFL', linewidth=4, markersize=1)
plt.plot(comm_accsfl, full_accuracies_smoothed4, linestyle='-', color='m', label='LocSplitFed', linewidth=4, markersize=1)



# Set labels and title with bold font
plt.xlabel('Communication Overhead (TB)', fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')

# Making the ticks bold and bigger
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(range(0, 101, 10), fontsize=14, fontweight='bold')

# Grid, legend, and save settings
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(fontsize=18)
plt.savefig("fmnist_accuracy_comm_overhead_exp1.pdf", format="pdf", bbox_inches="tight")

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