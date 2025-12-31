import csv
import itertools
import numpy as np
from itertools import permutations
import time

def load_devices_data(filename='devices_data.csv', subset_size=200):
    devices = []
    bandwidth_matrix = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        
        # Read only first `subset_size` rows
        for i, row in enumerate(reader):
            if i >= subset_size:
                break

            device_id = row[0]
            cpu_power = float(row[1])
            devices.append({'id': device_id, 'cpu_power': cpu_power})
            
            # Only take the first `subset_size` bandwidth values
            bandwidth_row = [float(x) if x else 0 for x in row[2:2 + subset_size]]
            bandwidth_matrix.append(bandwidth_row)

    print(f"Loaded {len(devices)} devices and their {subset_size}x{subset_size} bandwidth matrix.")
    return devices, bandwidth_matrix


from itertools import combinations, product
import numpy as np

import random

def find_all_assignments(devices, mylambda, max_assignments=1000000):
    N = len(devices)
    num_aggregators = round(mylambda * N)
    all_device_ids = [device['id'] for device in devices]

    all_assignments = []
    attempts = 0

    while len(all_assignments) < max_assignments and attempts < max_assignments * 10:
        attempts += 1

        # Randomly sample aggregator set
        aggregators = random.sample(all_device_ids, num_aggregators)
        weak_clients = [d for d in all_device_ids if d not in aggregators]

        # Random assignment of each weak client to a random aggregator
        assignment_matrix = np.zeros((N, N), dtype=int)
        for aggregator in aggregators:
            idx = all_device_ids.index(aggregator)
            assignment_matrix[idx][idx] = 1  # aggregator assigned to self

        for weak_client in weak_clients:
            assigned_agg = random.choice(aggregators)
            agg_idx = all_device_ids.index(assigned_agg)
            weak_idx = all_device_ids.index(weak_client)
            assignment_matrix[agg_idx][weak_idx] = 1

        all_assignments.append((aggregators, assignment_matrix))

    return all_assignments


# Example usage:
# devices, _ = load_devices_data()
# all_assignments = find_all_assignments(devices, mylambda)
# print(len(all_assignments))  # Total number of possible assignments
# for aggs, matrix in all_assignments[:5]:  # Print first 5 solutions
#     print(f"Aggregators: {aggs}")
#     print(matrix)

def compute_csfl_delay(h, v, V, layer_mflops_one_sample, layer_mbytes, bandwidth_matrix, devices, assignment_matrix, samples, p_s):
    """Computes the CSFL delay given the layer split (h, v), assignment matrix, and bandwidth constraints."""
    
    N = len(devices)
    aggregators = [i for i in range(N) if assignment_matrix[i][i] == 1]  # Aggregator indices
    weak_clients = [i for i in range(N) if assignment_matrix[i][i] == 0]  # Weak device indices
    
    # Compute the slowest weak-client to aggregator bandwidth (min bandwidth of assigned links)
    min_R_weak_agg = float('inf')
    for weak in weak_clients:
        assigned_agg = np.argmax(assignment_matrix[:, weak])  # Get assigned aggregator
        R_weak_agg = bandwidth_matrix[weak][assigned_agg]  # Get bandwidth between weak client & aggregator
        min_R_weak_agg = min(min_R_weak_agg, R_weak_agg)
    
    # Compute D0 using the slowest weak-aggregator link
    D0 = max(
        sum(layer_mbytes[:h]) / min_R_weak_agg,  
        sum(layer_mbytes[h:v]) / R_server
    )

    # Find the slowest weak-client to aggregator delay
    max_weak_agg_delay = 0
    for weak in weak_clients:
        assigned_agg = np.argmax(assignment_matrix[:, weak])  # Get assigned aggregator
        p_n = devices[weak]['cpu_power'] * 10**9  # Convert GHz to Hz
        R_weak_agg = bandwidth_matrix[weak][assigned_agg]  # Bandwidth for this weak-aggregator pair

        # Compute client-to-aggregator ratio (CAR) for this aggregator
        CAR = np.sum(assignment_matrix[assigned_agg]) - 1  # Exclude the aggregator itself
        
        term1_1 = (samples * sum(layer_mflops_one_sample[:h])) * 10**6 / p_n
        term1_2 = layer_mbytes[h+1] / R_weak_agg
        term1_3 = ((samples * sum(layer_mflops_one_sample[h:v])) * CAR) * 10**6 / p_k
        term1_4 = layer_mbytes[v] / min_R_weak_agg

        D1 = term1_1 + term1_2 + term1_3 + term1_4
        max_weak_agg_delay = max(max_weak_agg_delay, D1)

    term1 = ((2 * N * samples * sum(layer_mflops_one_sample[v:V+1]))) * 10**6 / p_s
    term2_1 = max_weak_agg_delay
    term2_2 = term1_4
    term2_3 = term1_1
    D2 = max(term1, term2_1 + term2_2 + term2_3)

    D3 = max(
        sum(layer_mbytes[:h]) / min_R_weak_agg,
        sum(layer_mbytes[h:v]) / min_R_weak_agg
    )

    D_round = D0 + (max_weak_agg_delay + D2) + D3
    return D_round

def find_optimal_assignment(devices, bandwidth_matrix, layer_mflops_one_sample, layer_mbytes, V, R_server, samples, p_k, p_s, mylambda):
    """Finds the optimal assignment matrix and (h, v) split that minimizes delay."""
    
    N = len(devices)
    min_delay = float('inf')
    best_h, best_v, best_assignment = None, None, None

    all_assignments = find_all_assignments(devices, mylambda, max_assignments=10000)

    for h in range(2, V):  
        for v in range(h + 1, V-1):  
            for aggregators, assignment_matrix in all_assignments:
                #(h, v, V, layer_mflops_one_sample, layer_mbytes, bandwidth_matrix, devices, assignment_matrix, samples, p_s):
                delay = compute_csfl_delay(h, v, V, layer_mflops_one_sample, layer_mbytes, bandwidth_matrix, devices, assignment_matrix, samples, p_s)

                if delay < min_delay:
                    min_delay = delay
                    best_h, best_v = h, v
                    best_assignment = assignment_matrix

    return best_h, best_v, min_delay, best_assignment

if __name__ == "__main__":
    # Load device data
    start_time = time.time()  # Start the timer
    devices, bandwidth_matrix = load_devices_data()

    layer_names=['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','fc1','fc2','fc3']
    layer_mflops_one_sample=[0.1,0.76,1.57,3.14,2.36,4.72,4.72,4.72,4.2,33.6,0.1]
    layer_mflops_one_sample = [i * 2 for i in layer_mflops_one_sample]
    #layer_mbytes=[10,10,10,10,10,10,10,10,10,10,10]
    layer_mbytes=[1,1,1,1,1,1,1,1,1,1,1]
    R_server = 5  # Mbps
    N = len(devices)
    samples = 60000 / N
    pk = 2.8 * 8 * 2
    ps = 100
    mylambda = 0.25
    p_k = pk * 10**9
    p_s = ps * 10**9
    V = 11

    optimal_h, optimal_v, min_delay, best_assignment = find_optimal_assignment(devices, bandwidth_matrix, layer_mflops_one_sample, layer_mbytes, V, R_server, samples, p_k, p_s, mylambda)

    print(f"Optimal (h, v): ({optimal_h}, {optimal_v}) with minimum CSFL delay: {min_delay:.6f} seconds")
    print("Optimal weak-client to aggregator assignment matrix:")
    print(best_assignment)
    end_time = time.time()  # End the timer
    duration = end_time - start_time
    print(f"Total execution time: {duration:.2f} seconds")