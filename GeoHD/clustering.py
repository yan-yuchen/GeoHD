import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Function to read a Shapefile and return a GeoDataFrame
def read_shapefile(shapefile_path):
    """
    Reads a Shapefile and returns a GeoDataFrame.

    Parameters:
    - shapefile_path (str): The path to the Shapefile directory.

    Returns:
    - GeoDataFrame: The GeoDataFrame containing the Shapefile data.
    """
    gdf = gpd.read_file(shapefile_path)
    return gdf

# Function to visualize points in the GeoDataFrame using Matplotlib
def visualize_points(gdf):
    """
    Visualizes the points in the GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to be visualized.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.plot(ax=ax, marker='o', markersize=5, color='blue', alpha=0.6)
    plt.title('Point Data Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Function to standardize features in the GeoDataFrame
def standardize_features(gdf):
    """
    Standardizes the features in the GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame containing the point data.

    Returns:
    - DataFrame: A DataFrame with standardized features.
    """
    features = gdf.drop(['geometry'], axis=1)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return pd.DataFrame(standardized_features, columns=features.columns)

# Function to perform PCA and reduce dimensionality
def perform_pca(standardized_features, n_components=2):
    """
    Performs Principal Component Analysis (PCA) to reduce dimensionality.

    Parameters:
    - standardized_features (array-like): The standardized features.
    - n_components (int, optional): The number of components to keep. Defaults to 2.

    Returns:
    - DataFrame: A DataFrame with the reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(standardized_features)
    return pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

# Function to perform t-SNE for further dimensionality reduction and visualization
def perform_tsne(pca_df, n_components=2):
    """
    Performs t-SNE to further reduce dimensionality and visualize the data.

    Parameters:
    - pca_df (DataFrame): The DataFrame with PCA results.
    - n_components (int, optional): The number of components to keep. Defaults to 2.

    Returns:
    - DataFrame: A DataFrame with the t-SNE results.
    """
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(pca_df)
    return pd.DataFrame(tsne_result, columns=[f't-SNE {i+1}' for i in range(n_components)])

# Function to perform clustering analysis using K-Means algorithm
def cluster_analysis(tsne_df, n_clusters=5):
    """
    Performs clustering analysis on the t-SNE DataFrame using K-Means algorithm.

    Parameters:
    - tsne_df (DataFrame): The DataFrame with t-SNE results.
    - n_clusters (int, optional): The number of clusters to form. Defaults to 5.

    Returns:
    - DataFrame: The DataFrame with additional columns for cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(tsne_df)
    return pd.concat([tsne_df, pd.DataFrame(kmeans_labels, columns=['Cluster'])], axis=1)

# Main function to execute the analysis pipeline
def main():
    # Shapefile path
    shapefile_path = 'path_to_your_shapefile.shp'  # Replace with your Shapefile path

    # Read Shapefile data
    gdf = read_shapefile(shapefile_path)

    # Visualize points from the GeoDataFrame
    visualize_points(gdf)

    # Standardize features in the GeoDataFrame
    standardized_features = standardize_features(gdf)

    # Perform PCA
    pca_df = perform_pca(standardized_features)

    # Visualize PCA results (if desired, can be skipped for a more concise visualization)
    # visualize_pca(pca_df)  # This function is not defined in the provided code, but can be created similarly to visualize_points

    # Perform t-SNE
    tsne_df = perform_tsne(pca_df)

    # Visualize t-SNE results
    visualize_tsne(tsne_df)

    # Perform clustering analysis
    clustered_df = cluster_analysis(tsne_df)

    # Visualize clustering results
    visualize_clusters(clustered_df)

# The main block is not defined in the provided code, so here is a placeholder for the visualization functions
def visualize_pca(pca_df):
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.title('PCA Results with Clustering')
    plt.show()

def visualize_tsne(tsne_df):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_df[0], tsne_df[1], c=tsne_df['Cluster'], cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(label='Cluster Label')
    plt.title('t-SNE Results')
    plt.show()

def visualize_clusters(clustered_df):
    plt.figure(figsize=(10, 8))
    plt.scatter(clustered_df[0], clustered_df[1], c=clustered_df['Cluster'], cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('t-SNE 1 (Clustered)')
    plt.ylabel('t-SNE 2 (Clustered)')
    plt.colorbar(label='Cluster Label')
    plt.title('Clustered t-SNE Results')
    plt.show()




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


# Function to evaluate the probability density of a multivariate Gaussian distribution
def multivariate_gaussian_pdf(point, mean_vector, covariance_matrix,multivariate_normal):
    """
    Evaluates the probability density of a multivariate Gaussian distribution at a given point.

    Parameters:
    - point (np.array): The point at which to evaluate the distribution.
    - mean_vector (np.array): The mean vector of the Gaussian distribution.
    - covariance_matrix (np.array): The covariance matrix of the Gaussian distribution.

    Returns:
    - float: The probability density value at the given point.
    """
    return multivariate_normal.pdf(point, mean=mean_vector, cov=covariance_matrix)

# Function to initialize the parameters for a set of Gaussian distributions
def initialize_gaussian_params(data_matrix, num_components):
    """
    Initializes the parameters (means, covariances, and weights) for a set of Gaussian distributions.

    Parameters:
    - data_matrix (np.array): The data points as a matrix.
    - num_components (int): The number of Gaussian distributions (clusters) to initialize.

    Returns:
    - tuple: A tuple containing the initialized means, covariances, and weights.
    """
    num_samples, num_features = data_matrix.shape
    chosen_indices = np.random.choice(num_samples, num_components, replace=False)
    initial_means = data_matrix[chosen_indices]
    initial_covariances = [np.cov(data_matrix.T) for _ in range(num_components)]
    initial_weights = np.full(num_components, 1 / num_components)
    return initial_means, initial_covariances, initial_weights

# Function to calculate the responsibilities (posteriors) for the EM algorithm
def calculate_responsibilities(data_matrix, gaussian_params):
    """
    Calculates the responsibilities (posteriors) for each data point belonging to each Gaussian distribution.

    Parameters:
    - data_matrix (np.array): The data points as a matrix.
    - gaussian_params (tuple): A tuple containing the means, covariances, and weights of the Gaussian distributions.

    Returns:
    - np.array: A matrix of responsibilities for each data point and Gaussian distribution.
    """
    means, covariances, weights = gaussian_params
    num_components = len(means)
    num_samples = data_matrix.shape[0]
    responsibilities = np.zeros((num_samples, num_components))
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        responsibilities[:, i] = weights[i] * multivariate_gaussian_pdf(data_matrix, mean, cov)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# Function to update the Gaussian parameters in the EM algorithm
def update_gaussian_params(data_matrix, responsibilities, gaussian_params):
    """
    Updates the Gaussian parameters (means, covariances, and weights) based on the responsibilities.

    Parameters:
    - data_matrix (np.array): The data points as a matrix.
    - responsibilities (np.array): The responsibilities (posteriors) for each data point and Gaussian distribution.
    - gaussian_params (tuple): The current parameters of the Gaussian distributions.

    Returns:
    - tuple: The updated Gaussian parameters.
    """
    means, covariances, weights = gaussian_params
    num_components = len(means)
    num_samples = data_matrix.shape[0]
    
    # Update parameters
    new_means = np.array([np.sum(responsibilities[:, i].reshape(-1, 1) * data_matrix) / np.sum(responsibilities[:, i]) for i in range(num_components)])
    new_covariances = [np.dot((data_matrix - new_means[i]).T, (responsibilities[:, i].reshape(-1, 1) * (data_matrix - new_means[i]))) for i in range(num_components)]
    new_weights = (np.sum(responsibilities[:, i]) / num_samples for i in range(num_components))
    
    return new_means, new_covariances, new_weights

# Function to perform the Expectation-Maximization (EM) algorithm for GMM clustering
def em_algorithm(data_matrix, num_components, max_iterations=100, convergence_threshold=1e-6):
    """
    Performs the Expectation-Maximization (EM) algorithm for Gaussian Mixture Model (GMM) clustering.

    Parameters:
    - data_matrix (np.array): The data points as a matrix.
    - num_components (int): The number of Gaussian distributions (clusters) to fit.
    - max_iterations (int): The maximum number of iterations to run the EM algorithm.
    - convergence_threshold (float): The convergence threshold to determine when to stop the EM algorithm.

    Returns:
    - tuple: A tuple containing the final means, covariances, and weights of the Gaussian distributions.
    """
    gaussian_params = initialize_gaussian_params(data_matrix, num_components)
    prev_log_likelihood = -np.inf
    log_likelihood = 0
    for iteration in range(max_iterations):
        # E-step: Calculate responsibilities
        responsibilities = calculate_responsibilities(data_matrix, gaussian_params)
        
        # M-step: Update Gaussian parameters
        gaussian_params = update_gaussian_params(data_matrix, responsibilities, gaussian_params)
        
        # Compute the log likelihood of the data
        log_likelihood = np.sum([np.log(np.sum(responsibilities[:, i] * multivariate_gaussian_pdf(data_matrix, gaussian_params[i][0], gaussian_params[i][1]))) for i in range(num_components)])
        
        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) < convergence_threshold:
            break
        prev_log_likelihood = log_likelihood
        
    return gaussian_params

# Function to assign cluster labels based on the responsibilities
def assign_clusters(data_matrix, gaussian_params):
    """
    Assigns cluster labels to each data point based on the highest responsibilities.

    Parameters:
    - data_matrix (np.array): The data points as a matrix.
    - gaussian_params (tuple): The final parameters of the Gaussian distributions.

    Returns:
    - np.array: An array of cluster labels for each data point.
    """
    means, _, _ = gaussian_params
    cluster_labels = np.argmax(calculate_responsibilities(data_matrix, gaussian_params), axis=1) + 1  # Add 1 to match the cluster labels with the plot
    return cluster_labels

# Function to visualize the GMM clustering results
def visualize_gmm_clusters(points_gdf, cluster_labels):
    """
    Visualizes the GMM clustering results on a map.

    Parameters:
    - points_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the point data.
    - cluster_labels (np.array): The cluster labels for each data point.

    Returns:
    - None: Displays a plot of the clusters.
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10, 10))
    for i, cluster_label in enumerate(np.unique(cluster_labels)):
        cluster_points = points_gdf[cluster_labels == cluster_label - 1]  # Subtract 1 to match the plot
        plt.scatter(cluster_points.geometry.x, cluster_points.geometry.y, c=colors[i % len(colors)], label=f'Cluster {cluster_label}')
    plt.legend()
    plt.title('GMM Clustering Results')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Main function to perform GMM clustering on point data from a shapefile
def perform_gmm_clustering(shapefile_path, num_clusters, max_iterations=100, convergence_threshold=1e-6):
    """
    Performs GMM clustering on point data from a shapefile and visualizes the results.

    Parameters:
    - shapefile_path (str): The path to the shapefile containing the point data.
    - num_clusters (int): The number of clusters to form.
    - max_iterations (int): The maximum number of iterations for the EM algorithm.
    - convergence_threshold (float): The convergence threshold for the EM algorithm.

    Returns:
    - None: Returns the cluster labels and displays a plot of the clusters.
    """
    points_gdf = gpd.read_file(shapefile_path)
    data_matrix = np.array([(point.x, point.y) for point in points_gdf.geometry])
    gaussian_params = em_algorithm(data_matrix, num_clusters, max_iterations, convergence_threshold)
    cluster_labels = assign_clusters(data_matrix, gaussian_params)
    visualize_gmm_clusters(points_gdf, cluster_labels)

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