# data_processing/clustering.py

from sklearn.cluster import KMeans

def cluster_features(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels
