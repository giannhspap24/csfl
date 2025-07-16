import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the data from CSV
data = pd.read_csv('devices_data.csv')

# Extract device IDs, CPU cycles, and connectivity bandwidth
device_ids = data['Device ID'].values
cpu_cycles = data['CPU Cycles (cycles/sec)'].values
bandwidth_data = data.iloc[:, 3:].values

# Replace NaN values with zeros for clustering purposes
bandwidth_data = np.nan_to_num(bandwidth_data)

# Calculate average bandwidth for each device
avg_bandwidth = np.mean(bandwidth_data, axis=1)

# Normalize CPU cycles and average bandwidth
scaler = MinMaxScaler()
cpu_cycles_normalized = scaler.fit_transform(cpu_cycles.reshape(-1, 1))
avg_bandwidth_normalized = scaler.fit_transform(avg_bandwidth.reshape(-1, 1))

# Combine normalized CPU cycles and average bandwidth into a single feature set
features = np.hstack((cpu_cycles_normalized, avg_bandwidth_normalized))

# Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
clusters = kmeans.labels_

# Find the best device in each cluster based on the combined score of CPU cycles and average bandwidth
best_devices = []
best_device_indices = []
for cluster in range(2):
    cluster_indices = np.where(clusters == cluster)[0]
    combined_score = cpu_cycles_normalized[cluster_indices] + avg_bandwidth_normalized[cluster_indices]
    best_device_index = cluster_indices[np.argmax(combined_score)]
    best_device_indices.append(best_device_index)
    best_devices.append(device_ids[best_device_index])

# Associate each device with the nearest best device
associations = []
for i, device_id in enumerate(device_ids):
    if i in best_device_indices:
        continue
    distances = [bandwidth_data[i][j] for j in best_device_indices]
    nearest_best_device_index = best_device_indices[np.argmax(distances)]
    associations.append((device_ids[nearest_best_device_index], device_id, bandwidth_data[i][nearest_best_device_index]))

# Create a graph
G = nx.Graph()

# Add nodes and edges with weights
for i, device_id in enumerate(device_ids):
    G.add_node(device_id)
    for j in range(i + 1, len(device_ids)):
        if not np.isnan(bandwidth_data[i][j]):
            G.add_edge(device_id, device_ids[j], weight=bandwidth_data[i][j])

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

# Highlight best devices
nx.draw_networkx_nodes(G, pos, nodelist=best_devices, node_size=700, node_color='green')

# Draw edges
edges = G.edges(data=True)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0, alpha=0.5)

# Draw edge labels
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12)

# Highlight associations
for best_device_id, device_id, bandwidth in associations:
    path_edges = [(device_id, best_device_id)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='red')
    path_label = f"{bandwidth:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(device_id, best_device_id): path_label}, font_color='red')

plt.title("Device Graph with Local-aggregators and Associations")
plt.show()

# Create a new CSV file with best device, associated device, and bandwidth connection
associations_df = pd.DataFrame(associations, columns=['Best Device', 'Associated Device', 'Bandwidth (Mbps)'])
associations_df.to_csv('device_associations.csv', index=False)

print("device_associations.csv has been created.")
