import random 
import csv
import pandas as pd
import numpy as np
from random import randrange, uniform
import time
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

import itertools

def assign_layers(devices, cpu_min, cpu_max):
    layer_count = 11
    server = 'Server'
    assignments = []

    strong_devices = [d['id'] for d in devices if d['cpu_power'] == cpu_max]
    weak_devices = [d['id'] for d in devices if d['cpu_power'] == cpu_min]

    for device in devices:
        layer_assignment = [None] * layer_count

        # Layer 1 always local
        layer_assignment[0] = device['id']

        if device['cpu_power'] == cpu_max:
            # Strong device trains layers 1–8 locally, 9–11 on server
            for i in range(1, 8):
                layer_assignment[i] = device['id']
            for i in range(8, 11):
                layer_assignment[i] = server

        else:
            # Weak device: 2 helpers max, sequential blocks
            # Divide layers 2–8 into up to 2 sequential blocks
            blocks = []
            block_sizes = []
            remaining = 7  # layers 2–8

            # Choose how to split: 1 full block or 2 blocks
            if random.random() < 0.5:
                # One block
                block_start = random.randint(1, 7)
                block_end = random.randint(block_start, 7)
                blocks = [(block_start, block_end)]
            else:
                # Two blocks
                split_point = random.randint(2, 6)
                blocks = [(1, split_point), (split_point + 1, 7)]

            # Assign helpers
            helpers = random.sample(strong_devices, k=min(len(blocks), len(strong_devices)))

            # Fill in layer assignment
            layer_assignment[0] = device['id']  # layer 1 always local
            for b, helper in zip(blocks, helpers):
                for i in range(b[0], b[1] + 1):
                    layer_assignment[i] = helper
            # Fill any unassigned layers with local device
            for i in range(1, 8):
                if layer_assignment[i] is None:
                    layer_assignment[i] = device['id']
            # Layers 9–11 on server
            for i in range(8, 11):
                layer_assignment[i] = server

        assignments.append(layer_assignment)

    return assignments, [d['id'] for d in devices]

def assign_layers_tirana(devices, layer_mflops):
    num_layers = len(layer_mflops)
    server_start = num_layers - last_layers
    assignment = {}
    
    # Separate strong and weak
    strong_devices = [d['id'] for d in devices if d['cpu_power'] == max(d['cpu_power'] for d in devices)]
    weak_devices = [d['id'] for d in devices if d['cpu_power'] != max(d['cpu_power'] for d in devices)]

    for device in devices:
        dev_id = device['id']
        assigned = [''] * num_layers

        if dev_id in strong_devices:
            for i in range(server_start):
                assigned[i] = dev_id
            for i in range(server_start, num_layers):
                assigned[i] = 'Server'
        
        else:
            # Weak device
            helper_candidates = random.sample(strong_devices, k=2)
            layer_idx = 0
            blocks = []

            # Assign initial local block (random size)
            local_block_size = random.randint(1, 4)
            local_block_end = min(server_start, layer_idx + local_block_size)
            blocks.append((dev_id, layer_idx, local_block_end))
            layer_idx = local_block_end

            # Remaining layers before server
            remaining = server_start - layer_idx
            if remaining > 0:
                # Split remaining into at most two sequential helper blocks
                split = random.randint(1, remaining) if remaining > 1 else remaining
                h1_end = layer_idx + split
                blocks.append((helper_candidates[0], layer_idx, h1_end))
                layer_idx = h1_end

                if layer_idx < server_start:
                    blocks.append((helper_candidates[1], layer_idx, server_start))

            # Fill the assignment
            for dev, start, end in blocks:
                for i in range(start, end):
                    assigned[i] = dev
            for i in range(server_start, num_layers):
                assigned[i] = 'Server'

        assignment[dev_id] = assigned

    return assignment

def assign_layers_iterative(devices, layer_mflops, v_range=(1, 4)):
    num_layers = len(layer_mflops)
    server_start = num_layers - 3
    assignment_by_k = {}

    # Sort devices by descending CPU power
    sorted_devices = sorted(devices, key=lambda d: d['cpu_power'], reverse=True)
    #max_k = len(devices) // 3
    max_k = 30

    for K in range(2,  max_k+1):
        helpers = [d['id'] for d in sorted_devices[:K]]
        weak_clients = [d for d in sorted_devices[K:]]

        for v in range(v_range[0], v_range[1] + 1):
            assignment = {}

            # Distribute weak clients across helpers (round robin)
            helper_idx = 0
            for dev in sorted_devices:
                dev_id = dev['id']
                assigned = [''] * num_layers

                if dev_id in helpers:
                    for i in range(server_start):
                        assigned[i] = dev_id
                    for i in range(server_start, num_layers):
                        assigned[i] = 'Server'
                else:
                    helper_id = helpers[helper_idx % K]
                    helper_idx += 1

                    # Assign layers:
                    # [0] local
                    # [1:v+1] helper
                    # [v+1:server_start] local again
                    assigned[0] = dev_id
                    for i in range(1, min(v + 1, server_start)):
                        assigned[i] = helper_id
                    for i in range(v + 1, server_start):
                        assigned[i] = dev_id
                    for i in range(server_start, num_layers):
                        assigned[i] = 'Server'

                assignment[dev_id] = assigned

            assignment_by_k[(K, v)] = assignment

    return assignment_by_k

def compute_delays_clients(devices, layer_assignments, layer_mflops, samples):
    delays = {}
    device_dict = {d['id']: d['cpu_power'] for d in devices}
    for dev_id, assigned_layers in layer_assignments.items():
        total_work = 0
        for idx, assigner in enumerate(assigned_layers):
            if assigner == dev_id:
                total_work += layer_mflops[idx]
        cpu_power = device_dict[dev_id]*10**9
        delay = (samples * total_work *10**6)  / (cpu_power)   # in microseconds
        delays[dev_id] = delay + 2*delay  #FOR FP AND BP.
    
    return delays

def compute_delay_server(num_devices,layer_mflops_one_sample, samples,v,V,p_s):
    delay_server=((2 * num_devices * samples*sum(layer_mflops_one_sample[v:V+1])))*10**6 / p_s
    return delay_server

def compute_delay_download(num_devices,layer_mbytes,v,V,R):
    D0 = max( sum(layer_mbytes[:v]) / R,  0 )
    return D0


    return comm_per_round_sfl

if __name__ == "__main__":
    num_devices = 100
    cpu_min = 1.5 * 2 *1
    cpu_max = 2.2 * 8 * 1 
    cpu_max=cpu_min *2
    bw_min = 20
    bw_max = 25

    R=bw_min
    mylambda = 0.3  # Fraction of devices with high CPU power
    samples=500
    v=8
    V=11
    ps=100 #Ghz
    p_s = ps * 10**9 

    devices = generate_devices_data(num_devices, cpu_min, cpu_max,mylambda)
    bandwidth_matrix = generate_bandwidth_data(devices, bw_min, bw_max)
    save_to_csv_devices(devices, bandwidth_matrix)
    
    #We work with MFLOPS
    layer_mflops_one_sample=[3.54,37,37,75,37,75,19,19,205,33,1] #for CIFAR-10 dataset
    #layer_mflops_one_sample=[1,7,7,5,2,1,1,1] #for MNIST dataset
    
    #We work with Mbytes
    layer_mbytes=[1,10,10,10,10,10,10,10,400,67,10] #for CIFAR-10 dataset
    #layer_mbytes=[1,7,8,7,10,80,30,10]#for MNIST dataset

    num_layers = len(layer_mflops_one_sample)
    last_layers = 3



    #delay_accSFL=delay_download+(slowest_client_delay/3)+max(delay_server, (2*slowest_client_delay/3) )+delay_download
    start_time = time.time()  # Start the timer
    #MY SOLUTION FROM DOWN and here
    print("-----My solution----")
    assignments = assign_layers_iterative(devices, layer_mflops_one_sample)
    end_time = time.time()  # End the timer
    duration = end_time - start_time
    print(f"Total execution time: {duration:.2f} seconds")
    print("-----My solution----")
    assignments = assign_layers_iterative(devices, layer_mflops_one_sample)

    best_Kv = None
    min_slowest_delay = float('inf')

    for (K, v), matrix in assignments.items():
        delays = compute_delays_clients(devices, matrix, layer_mflops_one_sample, samples)
        slowest_client_delay = max(delays.values())  # get the max delay among clients

        print(f"K={K}, v={v}, Slowest Client Delay: {slowest_client_delay:.2f} seconds")

        if slowest_client_delay < min_slowest_delay:
            min_slowest_delay = slowest_client_delay
            best_Kv = (K, v)

    print("\nBest configuration:")
    print(f"K={best_Kv[0]}, v={best_Kv[1]}, with slowest client delay = {min_slowest_delay:.2f} seconds")
    import matplotlib.pyplot as plt

    delays_list = []
    Ks = []
    for (K, v), matrix in assignments.items():
        delays = compute_delays_clients(devices, matrix, layer_mflops_one_sample, samples)
        slowest_client_delay = max(delays.values())
        Ks.append(K)
        delays_list.append(slowest_client_delay)

    plt.plot(Ks, delays_list, marker='o')
    plt.xlabel("Number of Helpers (K)")
    plt.ylabel("Slowest Client Delay (s)")
    plt.title("Effect of Helper Count on Worst Delay")
    plt.grid(True)
    plt.show()

