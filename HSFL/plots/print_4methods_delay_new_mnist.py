import matplotlib.pyplot as plt
import pandas as pd
import random
# Load Excel data
df_csfl = pd.read_excel("alexnet_MNIST_conv5_100clients_10_agg_iid_300_epochs.xlsx")
df_sfl_tirana = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_sfl = pd.read_excel("sfl_100_clients_fed_avg_freq_3_cut_layer_conv5.xlsx")
df_accsfl = pd.read_excel("alexnet_MNIST_conv5_100clients_100_agg_iid_300_epochs.xlsx")#replace with 80aggregators

# Extract epoch and accuracy
epochs_csfl, acc_csfl = df_csfl['round'], df_csfl['acc_test']
epochs_sfl_tirana, acc_sfl_tirana = df_sfl_tirana['round'], df_sfl_tirana['acc_test']
epochs_sfl, acc_sfl = df_sfl['round'], df_sfl['acc_test']
epochs_accsfl, acc_accsfl = df_accsfl['round'], df_accsfl['acc_test']

# Define time per epoch in seconds
time_per_epoch_csfl = 7.3
time_per_epoch_sfl_tirana = 10.2
time_per_epoch_sfl = 12
time_per_epoch_accsfl = 8.6

# Convert epoch to cumulative delay (in seconds)
time_csfl = epochs_csfl * time_per_epoch_csfl
time_sfl_tirana = epochs_sfl_tirana * time_per_epoch_sfl_tirana
time_sfl = epochs_sfl * time_per_epoch_sfl
time_accsfl = epochs_accsfl * time_per_epoch_accsfl

# Optional: apply smoothing (e.g. moving average)
def smooth(data, window=10):
    return data.rolling(window=window, min_periods=1).mean()

acc_csfl = smooth(acc_csfl)
acc_sfl_tirana = smooth(acc_sfl_tirana)
acc_sfl = smooth(acc_sfl)
acc_accsfl = smooth(acc_accsfl)
# Maximum time to show (in seconds)
max_time = 2000

# Filter all data up to max_time
mask_csfl = time_csfl <= max_time
time_csfl = time_csfl[mask_csfl]
acc_csfl = acc_csfl[mask_csfl]

mask_sfl_tirana = time_sfl_tirana <= max_time
time_sfl_tirana = time_sfl_tirana[mask_sfl_tirana]
acc_sfl_tirana = acc_sfl_tirana[mask_sfl_tirana]

mask_sfl = time_sfl <= max_time
time_sfl = time_sfl[mask_sfl]
acc_sfl = acc_sfl[mask_sfl]

mask_accsfl = time_accsfl <= max_time
time_accsfl = time_accsfl[mask_accsfl]
acc_accsfl = acc_accsfl[mask_accsfl]



# Plotting the lines

plt.plot(time_csfl, acc_csfl,linestyle='--', color='c', label='SFL3', linewidth=4,markersize=1)
plt.plot(time_accsfl, acc_accsfl,  linestyle='-', color='m', label='LocSFL',  linewidth=4, markersize=1)
plt.plot(time_sfl_tirana, acc_sfl_tirana,  linestyle='dashdot', color='g', label='Multihop SFL', linewidth=4,markersize=1)
plt.plot(time_sfl, acc_sfl,linestyle='dotted', color='b', label='SFL', linewidth=4,markersize=1)


# Labels
plt.xlabel('Training Delay (seconds)', fontsize=18, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=18, fontweight='bold')

# Set custom Y-ticks
plt.yticks(range(0, 101, 10), fontsize=14, fontweight='bold')
plt.xticks(range(0, 2001, 400),fontsize=14, fontweight='bold')

# Shrink margins so lines touch the border
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(fontsize=18)

# Place legend outside if needed (optional)
# plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

# Save and show
plt.savefig("accuracy_vs_delay_mnist.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Calculate and print the highest smoothed accuracies within the time range
max_full_accuracy1 = max(acc_csfl)
max_full_accuracy2 = max(acc_sfl)
max_full_accuracy3 = max(acc_accsfl)

print("\nHighest Smoothed Full Accuracies:")
print(f"CSFL: {max_full_accuracy2:.2f}%")
print(f"SFL: {max_full_accuracy2:.2f}%")
print(f"AccSFL: {max_full_accuracy3:.2f}%")