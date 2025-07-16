import random 
import csv
import pandas as pd
import numpy as np
from random import randrange, uniform
#Random power within the range [cpu_min, cpu_max]
# def generate_devices_data(num_devices, cpu_min, cpu_max):
#     devices = []
#     for i in range(num_devices):
#         device_id = f'Device_{i+1}'
#         cpu_cycles = uniform(cpu_min, cpu_max)  # in Ghz ()
#         devices.append({
#             'id': device_id,
#             'cpu_power': cpu_cycles,
#         })
#     return devices

#Random power either cpu_min or cpu_max
def generate_devices_data(num_devices, cpu_min, cpu_max, mylambda):
    devices = []
    num_high_power = int(num_devices * mylambda)  # Devices with cpu_max
    num_low_power = num_devices - num_high_power  # Devices with cpu_min

    for i in range(num_high_power):
        devices.append({'id': f'Device_{i+1}', 'cpu_power': cpu_max})

    for i in range(num_high_power, num_devices):
        devices.append({'id': f'Device_{i+1}', 'cpu_power': cpu_min})

   # random.shuffle(devices)  # Shuffle to mix low and high power devices
    return devices

def generate_bandwidth_data(devices, bw_min, bw_max):
    num_devices = len(devices)
    bandwidth_matrix = [[None] * num_devices for _ in range(num_devices)]
    for i in range(num_devices):
        for j in range(i + 1, num_devices):
            if random.choice([True, True]):  # randomly decide if there is a connection --> There is connection for sure
                bandwidth = round(random.uniform(bw_min, bw_max), 2)  # in Mbps
                bandwidth_matrix[i][j] = bandwidth
                bandwidth_matrix[j][i] = bandwidth
    return bandwidth_matrix

def save_to_csv_devices(devices, bandwidth_matrix, filename='devices_data.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Device_id', 'CPU_power'] + 
                        [f'Link to {device["id"]} (Mbps)' for device in devices])
        
        # Write the device information and bandwidth data
        for i, device in enumerate(devices):
            row = [device['id'], device['cpu_power']] + bandwidth_matrix[i]
            writer.writerow(row)
    print(f"Data for {len(devices)} devices has been saved to '{filename}'.")


if __name__ == "__main__":
    num_devices = 10
    cpu_min = 1.5 * 4 *2
    cpu_max = 2.8 * 8 * 2 
    bw_min = 1
    bw_max = 10
    mylambda = 0.3  # Fraction of devices with high CPU power

    devices = generate_devices_data(num_devices, cpu_min, cpu_max,mylambda)
    bandwidth_matrix = generate_bandwidth_data(devices, bw_min, bw_max)
    save_to_csv_devices(devices, bandwidth_matrix)
    
    # Generate model.csv for VGG-11 layers
    #vgg_11_layers = [
       # "conv1_1", "conv2_1", "conv3_1", "conv3_2",
      # "conv4_1", "conv4_2", "conv5_1", "conv5_2",
     #   "fc1", "fc2", "fc3"
    #]
   # generate_model_csv(vgg_11_layers)
