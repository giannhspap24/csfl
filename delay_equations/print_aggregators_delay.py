import matplotlib.pyplot as plt
import pandas as pd
import random
# Load Excel data
df_csfl = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")
df_sfl_tirana = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")
df_sfl = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")
#df_accsfl = pd.read_excel("alexnet_MNIST_conv5_100clients_5agg_iid.xlsx")

# Extract epoch and accuracy
epochs_csfl, acc_csfl = df_csfl['round'], df_csfl['acc_test']
epochs_sfl_tirana, acc_sfl_tirana = df_sfl_tirana['round'], df_sfl_tirana['acc_test']
epochs_sfl, acc_sfl = df_sfl['round'], df_sfl['acc_test']


# Define time per epoch in seconds
time_per_epoch_csfl = 7.3
time_per_epoch_sfl_tirana = 7.8
time_per_epoch_sfl = 8


# Convert epoch to cumulative delay (in seconds)
time_csfl = epochs_csfl * time_per_epoch_csfl
time_sfl_tirana = epochs_sfl_tirana * time_per_epoch_sfl_tirana
time_sfl = epochs_sfl * time_per_epoch_sfl


# Optional: apply smoothing (e.g. moving average)
def smooth(data, window=10):
    return data.rolling(window=window, min_periods=1).mean()

acc_csfl = smooth(acc_csfl)
acc_sfl_tirana = smooth(acc_sfl_tirana)
acc_sfl = smooth(acc_sfl)

# Maximum time to show (in seconds)
max_time = 1460

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


plt.figure(figsize=(10, 6))  # You can adjust size if needed

# Plotting the lines
plt.plot(time_csfl, acc_csfl, linestyle='--', color='c', label='HSFL-ll(λ=0.05)', linewidth=3)
plt.plot(time_sfl_tirana, acc_sfl_tirana, linestyle='-', color='g', label='HSFL-ll(λ=0.15)', linewidth=3)
plt.plot(time_sfl, acc_sfl, linestyle='-', color='b', label='HSFL-ll(λ=0.3)', linewidth=3)

# Labels
plt.xlabel('Training Delay (seconds)', fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')

# Set custom Y-ticks
plt.yticks(range(0, 101, 10), fontsize=13, fontweight='bold')
plt.xticks(fontsize=13, fontweight='bold')

# Shrink margins so lines touch the border
plt.margins(x=0)  # No horizontal margin
plt.grid(True, linestyle='--', linewidth=0.5)

# Place legend outside if needed (optional)
# plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(fontsize=14, loc='lower right')

# Tighten layout
plt.tight_layout(pad=0.5)

# Save and show
plt.savefig("accuracy_vs_delay_upto_1450s.pdf", format="pdf", bbox_inches="tight")
plt.show()
