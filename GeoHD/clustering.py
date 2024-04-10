import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


# Helper function to calculate Euclidean distance
def calculate_euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to initialize centroids for K-Means clustering
def initialize_kmeans_centroids(data_matrix, num_clusters, random_seed=None):
    """
    Initialize centroids for K-Means clustering by randomly selecting data points.
    """
    np.random.seed(random_seed)
    indices = np.random.choice(data_matrix.shape[0], size=num_clusters, replace=False)
    centroids = data_matrix[indices]
    return centroids

# Function to assign clusters to points in K-Means clustering
def assign_kmeans_clusters(data_matrix, centroids):
    """
    Assign each data point to the nearest centroid to determine its cluster.
    """
    clusters = []
    for point in data_matrix:
        distances = [calculate_euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters.append(closest_centroid_index)
    return np.array(clusters)

# Function to update centroids in K-Means clustering
def update_kmeans_centroids(data_matrix, clusters, num_clusters):
    """
    Update the centroids based on the current cluster assignments.
    """
    new_centroids = np.zeros((num_clusters, data_matrix.shape[1]))
    for cluster_index in range(num_clusters):
        cluster_points = data_matrix[clusters == cluster_index]
        if len(cluster_points) > 0:
            new_centroids[cluster_index] = np.mean(cluster_points, axis=0)
    return new_centroids

# Function to visualize K-Means clustering results
def visualize_kmeans_clusters(data_matrix, clusters, centroids):
    """
    Visualize the clusters and centroids after K-Means clustering.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=150, edgecolor='k')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('K-Means Clustering Visualization')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

# Main function to perform K-Means clustering
def perform_kmeans_clustering(data_matrix, num_clusters, max_iterations, random_seed=None):
    """
    Perform K-Means clustering on a given dataset.
    """
    centroids = initialize_kmeans_centroids(data_matrix, num_clusters, random_seed)
    for _ in range(max_iterations):
        clusters = assign_kmeans_clusters(data_matrix, centroids)
        new_centroids = update_kmeans_centroids(data_matrix, clusters, num_clusters)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    visualize_kmeans_clusters(data_matrix, clusters, centroids)
    return clusters, centroids

# Helper function to find neighbors within a specified radius
def find_neighbors(data_matrix, point_index, radius):
    """
    Find all points within a specified radius of a given point.
    """
    distances = pairwise_distances(data_matrix, [data_matrix[point_index]])
    neighbors = np.where(distances[0] <= radius)[1]
    return neighbors

# Function to expand a cluster based on DBSCAN algorithm
def expand_cluster(data_matrix, labels, point_index, radius, min_samples,neighbors):
    """
    Expand a cluster by including all directly密度 reachable points.
    """
    labels[point_index] = -1  # Mark as visited
    if len(neighbors) < min_samples:
        return  # Not a core point

    # Assign cluster label and expand cluster
    cluster_id = len(labels)
    labels[point_index] = cluster_id
    for neighbor in neighbors:
        if labels[neighbor] == 0:
            expand_cluster(data_matrix, labels, neighbor, radius, min_samples)

# Main function to perform DBSCAN clustering
def perform_dbscan_clustering(data_matrix, radius, min_samples):
    """
    Perform DBSCAN clustering on a given dataset.
    """
    labels = np.zeros(len(data_matrix), dtype=int)
    clusters = []
    for i in range(len(data_matrix)):
        if labels[i] == 0:
            neighbors = find_neighbors(data_matrix, i, radius)
            if len(neighbors) < min_samples:
                labels[i] = -1  # Mark as noise
            else:
                expand_cluster(data_matrix, labels, i, radius, min_samples)
                cluster_id = len(clusters)
                clusters.append(cluster_id)

    return labels, clusters

# Visualization of DBSCAN clusters (assuming 2D data for simplicity)
def visualize_dbscan_clusters(data_matrix, labels):
    """
    Visualize the DBSCAN clustering results.
    """
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            plt.scatter(data_matrix[labels == cluster_id, 0], data_matrix[labels == cluster_id, 1], color='black', label='Noise')
        else:
            plt.scatter(data_matrix[labels == cluster_id, 0], data_matrix[labels == cluster_id, 1], label=f'Cluster {cluster_id}')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('DBSCAN Clustering Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to create a grid over the spatial extent of the points
def create_grid(xmin, ymin, xmax, ymax, grid_size):
    """
    Create a grid of cells over the spatial extent of the points.
    """
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)
    grid = [(x, y) for x in xgrid for y in ygrid]
    return grid

# Function to calculate statistics for each grid cell
def calculate_grid_statistics(points, grid,grid_size):
    """
    Calculate statistics for each grid cell based on the points within it.
    """
    statistics = {cell: 0 for cell in grid}
    for point in points:
        for cell in grid:
            if point[0] >= cell[0] and point[0] < cell[0] + grid_size and point[1] >= cell[1] and point[1] < cell[1] + grid_size:
                statistics[cell] += 1
    return statistics

# Main function to perform STING clustering
def perform_sting_clustering(points, grid_size):
    """
    Perform STING clustering on a given set of points.
    """
    grid_statistics = calculate_grid_statistics(points, create_grid(*points.total_bounds, grid_size))
    dense_cells = {cell: count for cell, count in grid_statistics.items() if count > np.mean(list(grid_statistics.values())) + np.std(list(grid_statistics.values()))}

    # Assign points to clusters based on dense cells
    clusters = {cell: [] for cell in dense_cells}
    for point in points:
        for cell, count in dense_cells.items():
            if point.x >= cell[0] and point.x < cell[0] + grid_size and point.y >= cell[1] and point.y < cell[1] + grid_size:
                clusters[cell].append(point)

    # Plot STING clusters
    def plot_sting_clusters(ax, points, clusters):
        for cell, points_in_cell in clusters.items():
            ax.scatter([point.x for point in points_in_cell], [point.y for point in points_in_cell], label=f'Cell {cell[0]} to {cell[0]+grid_size}')
        ax.set_xlabel('X-Coordinate')
        ax.set_ylabel('Y-Coordinate')
        ax.set_title('STING Clustering Visualization')
        ax.legend()
        ax.grid(True)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plot_sting_clusters(ax, points, clusters)
    plt.show()

from sklearn.neighbors import NearestNeighbors

# Function to calculate reachability distances for OPTICS
def calculate_optics_distances(data_matrix, point_index, neighbors, min_samples, reachability_distance):
    """
    Calculate reachability distances for OPTICS based on the nearest neighbors.
    """
    distances = [calculate_euclidean_distance(data_matrix[point_index], data_matrix[neighbor_index]) for neighbor_index in neighbors]
    core_distance = max(distances) if len(neighbors) >= min_samples else 0
    reachability_distance[point_index] = core_distance if core_distance > reachability_distance[point_index] else reachability_distance[point_index]

# Main function to perform OPTICS clustering
def perform_optics_clustering(data_matrix, epsilon, min_samples):
    """
    Perform OPTICS clustering on a given dataset.
    """
    num_points = data_matrix.shape[0]
    reachability_distances = np.full(num_points, np.inf)
    cluster_labels = np.full(num_points, -1)

    # Find nearest neighbors for each point
    nearest_neighbors = NearestNeighbors(n_neighbors=min_samples + 1).fit(data_matrix)
    distances, indices = nearest_neighbors.kneighbors(data_matrix)

    # Calculate reachability distances and assign clusters
    for i in range(num_points):
        neighbors = indices[i][1:]  # Exclude the point itself
        calculate_optics_distances(data_matrix, i, neighbors, min_samples, reachability_distances)
        if len(neighbors) >= min_samples:
            cluster_id = i
            while True:
                cluster_labels[i] = cluster_id
                neighbors = [j for j in range(num_points) if calculate_euclidean_distance(data_matrix[i], data_matrix[j]) <= epsilon and j != i]
                if len(neighbors) == 0:
                    break
                # Find next core point with minimum reachability distance
                next_core = min(neighbors, key=lambda x: reachability_distances[x])
                if reachability_distances[next_core] > epsilon:
                    cluster_labels[next_core] = -1  # Mark as noise
                else:
                    cluster_id += 1
                    cluster_labels[next_core] = cluster_id
                    break

    return cluster_labels

# Visualization of OPTICS clusters (assuming 2D data for simplicity)
def visualize_optics_clusters(data_matrix, cluster_labels):
    """
    Visualize the OPTICS clustering results.
    """
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            plt.scatter(data_matrix[cluster_labels == cluster_id, 0], data_matrix[cluster_labels == cluster_id, 1], color='black', label='Noise')
        else:
            plt.scatter(data_matrix[cluster_labels == cluster_id, 0], data_matrix[cluster_labels == cluster_id, 1], label=f'Cluster {cluster_id}')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('OPTICS Clustering Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to calculate pairwise distances between points
def calculate_distances(data_matrix,distance):
    """
    Calculate all pairwise distances between points in a dataset.
    """
    return distance.squareform(distance.pdist(data_matrix, 'euclidean'))

# Function to merge clusters based on the closest distance
def merge_closest_clusters(distances, clusters, max_clusters):
    """
    Merge clusters based on the minimum distance between them.
    """
    num_clusters = len(clusters)
    while num_clusters > max_clusters:
        min_dist_index = np.argmin(distances)
        # Find the two clusters involved in the minimum distance
        cluster1, cluster2 = None, None
        for i, cluster in enumerate(clusters):
            if min_dist_index in cluster:
                cluster1 = i
        for i, cluster in enumerate(clusters):
            if min_dist_index + 1 in cluster:
                cluster2 = i
        # Merge the two clusters
        clusters[cluster1] = list(set(clusters[cluster1]) | set(clusters[cluster2]))
        clusters[cluster2] = []
        distances = np.delete(distances, cluster2, axis=0)
        distances = np.delete(distances, cluster2, axis=1)
        num_clusters -= 1
    return clusters

# Function to perform hierarchical clustering
def perform_hierarchical_clustering(data_matrix, max_clusters):
    """
    Perform hierarchical clustering on a given dataset.
    """
    distances = calculate_distances(data_matrix)
    clusters = list(range(len(data_matrix)))
    merge_closest_clusters(distances, clusters, max_clusters)
    return clusters

# Visualization of hierarchical clusters (assuming 2D data for simplicity)
def visualize_hierarchical_clusters(data_matrix, cluster_labels):
    """
    Visualize the hierarchical clustering results.
    """
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(cluster_labels):
        cluster_points = data_matrix[cluster_labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('Hierarchical Clustering Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()