# active_learning/sample_selection.py

import numpy as np

def select_samples_by_uncertainty(uncertainties, data, k):
    indices = np.argsort(uncertainties)[-k:]  # Select top k uncertain samples
    selected_data = [data[i] for i in indices]
    return selected_data

def select_samples_by_clustering(cluster_labels, data, samples_per_cluster):
    selected_data = []
    clusters = np.unique(cluster_labels)
    for cluster in clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        selected_indices = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
        selected_data.extend([data[i] for i in selected_indices])
    return selected_data
