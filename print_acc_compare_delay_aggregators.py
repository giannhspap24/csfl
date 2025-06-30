import matplotlib.pyplot as plt
import pandas as pd
import random



# Load the datasets
#df1 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv3_4clients_1agg_iid.xlsx")
#df2=pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv4_4clients_1agg_iid.xlsx")
#df3 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv5_4clients_1agg_iid.xlsx")

#df1 = pd.read_excel("AlexNet-MNIST-ALL-EXPS-IID/alexnet_MNIST_conv2_4clients_1agg_iid.xlsx")
#df2=pd.read_excel("AlexNet-MNIST-ALL-EXPS-IID/alexnet_MNIST_conv4_4clients_1agg_iid.xlsx")
#df3 = pd.read_excel("AlexNet-MNIST-ALL-EXPS-IID/alexnet_MNIST_conv5_4clients_1agg_iid.xlsx")

#df1 = pd.read_excel("MNIST-FMNIST-NON-IID-ALEXNET-ACCSFL-SFLV1/accsfl_alexnet_FMNIST_conv3_100clients_iid.xlsx")
#df2=pd.read_excel("MNIST-FMNIST-NON-IID-ALEXNET-ACCSFL-SFLV1/accsfl_alexnet_FMNIST_conv4_100clients_iid.xlsx")
#df3 = pd.read_excel("MNIST-FMNIST-NON-IID-ALEXNET-ACCSFL-SFLV1/accsfl_alexnet_FMNIST_conv5_100clients_iid.xlsx")


#df1 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv3_100clients_2agg_iid.xlsx")
#df2=pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv4_100clients_2agg_iid.xlsx")
#df3 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv5_100clients_2agg_iid.xlsx")

#df1 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv2_100clients_2agg_iid.xlsx")
#df2=pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv3_100clients_2agg_iid.xlsx")
#df3 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv4_100clients_2agg_iid.xlsx")
#df4 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/alexnet_FMNIST_conv5_100clients_2agg_iid.xlsx")

df1 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/sflv1_alexnet_MNIST_conv2_100clients_iid.xlsx")
df2=pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/sflv1_alexnet_MNIST_conv3_100clients_iid.xlsx")
df3 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/sflv1_alexnet_MNIST_conv4_100clients_iid.xlsx")
df4 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID/sflv1_alexnet_MNIST_conv5_100clients_iid.xlsx")
#df5 = pd.read_excel("AlexNet-FMNIST-ALL-EXPS-IID//sflv1_alexnet_FMNIST_conv5_100clients_iid_new.xlsx")
#C:\Users\John\Downloads\data_csfl\MNIST-FMNIST-IID-ALEXNET-ACCSFL-SFLV1
# Extract full accuracies and epochs
full_accuracies1, epochs1 = df1['acc_test'], df1['round']
full_accuracies2, epochs2 = df2['acc_test'], df2['round']
full_accuracies3, epochs3 = df3['acc_test'], df3['round']
full_accuracies4, epochs4 = df4['acc_test'], df4['round']
#full_accuracies5, epochs5 = df5['acc_test'], df5['round']
# Time per epoch (in seconds)
#time_per_epoch_csfl = 9  #10.2,9
#time_per_epoch_sfl = 18     #
#time_per_epoch_accsfl = 14  #

# Time per epoch (in seconds)
time_per_epoch_csfl = 1  #10.2,9 # 105
time_per_epoch_sfl = 1    # 8   #132
time_per_epoch_accsfl = 1 # 7   # 122
time_per_epoch_accsfl1 = 1
time_per_epoch_sfl1 = 1
# Maximum time to plot (443 * 9 seconds)
#max_time = time_per_epoch_csfl*200
max_time = 200
# Calculate cumulative time for each method
time_csfl = epochs1 * time_per_epoch_csfl
time_sfl = epochs2 * time_per_epoch_sfl
time_accsfl = epochs3 * time_per_epoch_accsfl
time_accsfl1 = epochs4 * time_per_epoch_accsfl1
#time_sfl1 = epochs5 * time_per_epoch_sfl1

# Add initial time = 0, accuracy = 0 for each method
time_csfl = pd.concat([pd.Series([0]), time_csfl], ignore_index=True)
full_accuracies1 = pd.concat([pd.Series([0]), full_accuracies1], ignore_index=True)
 
time_sfl = pd.concat([pd.Series([0]), time_sfl], ignore_index=True)
full_accuracies2 = pd.concat([pd.Series([0]), full_accuracies2], ignore_index=True)

time_accsfl = pd.concat([pd.Series([0]), time_accsfl], ignore_index=True)
full_accuracies3 = pd.concat([pd.Series([0]), full_accuracies3], ignore_index=True)

time_accsfl1 = pd.concat([pd.Series([0]), time_accsfl1], ignore_index=True)
full_accuracies4 = pd.concat([pd.Series([0]), full_accuracies4], ignore_index=True)


#time_sfl1 = pd.concat([pd.Series([0]), time_sfl1], ignore_index=True)
#full_accuracies5 = pd.concat([pd.Series([0]), full_accuracies5], ignore_index=True)

# Restrict time to the maximum time range
time_csfl = time_csfl[time_csfl <= max_time]
time_sfl = time_sfl[time_sfl <= max_time]
time_accsfl = time_accsfl[time_accsfl <= max_time]
time_accsfl1 = time_accsfl1[time_accsfl1 <= max_time]
#time_sfl1 = time_sfl1[time_sfl1 <= max_time]
# Truncate accuracies to align with the time limits
full_accuracies1 = full_accuracies1[:len(time_csfl)]
full_accuracies2 = full_accuracies2[:len(time_sfl)]
full_accuracies3 = full_accuracies3[:len(time_accsfl)]
full_accuracies4 = full_accuracies4[:len(time_accsfl1)]
#full_accuracies5 = full_accuracies5[:len(time_sfl1)]
# Function to calculate the moving average
def calculate_moving_average(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean().reset_index(drop=True)

# Apply moving average smoothing
full_accuracies_smoothed1 = calculate_moving_average(full_accuracies1)
full_accuracies_smoothed2 = calculate_moving_average(full_accuracies2)
full_accuracies_smoothed3 = calculate_moving_average(full_accuracies3)
full_accuracies_smoothed4 = calculate_moving_average(full_accuracies4)
#full_accuracies_smoothed5 = calculate_moving_average(full_accuracies5)
# Plot accuracies over time
#plt.figure(figsize=(12, 7))
#plt.plot(time_csfl, full_accuracies_smoothed1,  linestyle='-', color='r', label='CSFL', linewidth=2)
#plt.plot(time_sfl, full_accuracies_smoothed2,  linestyle='--', color='b', label='SFL', linewidth=2)
#plt.plot(time_accsfl, full_accuracies_smoothed3, linestyle='-.', color='g', label='AccSFL', linewidth=2)

# More distinguishable line styles and thicker lines
plt.plot(time_csfl, full_accuracies_smoothed1, linestyle='-', marker='*', color='r', label='SFL(Conv 2)', linewidth=4, markersize=1)
plt.plot(time_accsfl, full_accuracies_smoothed2, linestyle='-.', marker='o', color='g', label='SFL(Conv 3)', linewidth=4, markersize=1)
plt.plot(time_sfl, full_accuracies_smoothed3, linestyle='--', marker='s', color='b', label='SFL(Conv 4)', linewidth=4, markersize=1)
plt.plot(time_sfl, full_accuracies_smoothed4, linestyle='-', marker='*', color='y', label='SFL(Conv 5)', linewidth=4, markersize=1)
#plt.plot(time_sfl, full_accuracies_smoothed5, linestyle='-', marker='s', color='m', label='SFL(Conv 5)', linewidth=4, markersize=1)
# Set labels and title with bold font
plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')

# Making the ticks bold and bigger
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(range(0, 101, 10), fontsize=14, fontweight='bold')

# Grid, legend, and save settings
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(fontsize=18,loc='lower right')

plt.savefig("vgg11_accuracy_train_delay_exp1.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

# Calculate and print the highest smoothed accuracies within the time range
max_full_accuracy1 = max(full_accuracies_smoothed1)
max_full_accuracy2 = max(full_accuracies_smoothed2)
max_full_accuracy3 = max(full_accuracies_smoothed3)
max_full_accuracy4 = max(full_accuracies_smoothed4)

print("\nHighest Smoothed Full Accuracies:")
print(f"CSFL: {max_full_accuracy1:.2f}%")
print(f"SFL: {max_full_accuracy2:.2f}%")
print(f"AccSFL: {max_full_accuracy3:.2f}%")
print(f"AccSFL: {max_full_accuracy4:.2f}%")