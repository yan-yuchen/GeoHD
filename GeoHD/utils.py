import numpy as np
import warnings
import random
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

def extract_hotspots(density_data_path, window_size=10, threshold=0):
    """
    Extract hotspots from density data using a window analysis approach.

    Parameters:
        density_data_path (str): Path to the .npy file containing density data.
        window_size (int): Size of the window for maximum filter (default is 10).
        threshold (int): Threshold value for extreme regions (default is 0).

    Returns:
        hotspots (ndarray): Array containing coordinates of hotspots.
    """
    # Load density data from the saved .npy file
    density_data = np.load(density_data_path)

    # Step 1: Window analysis to get maximum values within each window
    max_density_surface = maximum_filter(density_data, size=window_size)

    # Step 2: Algebraic subtraction to get non-negative surface
    non_negative_surface = max_density_surface - density_data

    # Step 3: Reclassification algorithm to classify into extreme and non-extreme regions
    extreme_region = non_negative_surface == threshold

    # Step 4: Extract hotspots from extreme regions
    hotspots = np.argwhere(extreme_region)

    return hotspots

def visualize_hotspots(density_data, hotspots):
    """
    Visualize hotspots on the density data.

    Parameters:
        density_data (ndarray): Density data array.
        hotspots (ndarray): Array containing coordinates of hotspots.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(density_data, cmap='Purples', origin='lower')
    plt.colorbar(label='Density')
    plt.scatter(hotspots[:, 1], hotspots[:, 0], color='red', s=5, label='Hotspots')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hotspot Extraction')
    plt.legend()
    plt.show()

def spectral_clustering_with_plot(points_shapefile, num_clusters=3, k=5, sigma=1):
    """
    Perform spectral clustering on a set of points from a shapefile and plot the clusters.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    num_clusters (int): Number of clusters to partition the data into.
    k (int): Number of nearest neighbors to consider for constructing the adjacency matrix.
    sigma (float): Sigma parameter for the Gaussian similarity function.

    Returns:
    np.array: Cluster labels for each point.
    """

    def gaussian_similarity_matrix(points, sigma=1):
        n = len(points)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = np.exp(-((points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2) / (2 * sigma ** 2))
        return similarity_matrix

    def adjacency_matrix(similarity_matrix, k):
        n = len(similarity_matrix)
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            idx = np.argsort(similarity_matrix[i])[::-1][:k]
            adjacency_matrix[i][idx] = similarity_matrix[i][idx]
        return adjacency_matrix

    def laplacian_matrix(adjacency_matrix):
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        laplacian_matrix = degree_matrix - adjacency_matrix
        return laplacian_matrix

    def compute_eigenvectors(laplacian_matrix, num_eigenvectors):
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return sorted_eigenvalues, sorted_eigenvectors[:, :num_eigenvectors]

    def kmeans_clustering(data, k, max_iters=100):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        for _ in range(max_iters):
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return labels

    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)

    # Compute similarity matrix
    similarity_matrix = gaussian_similarity_matrix(points_gdf.geometry, sigma)

    # Construct adjacency matrix
    adj_matrix = adjacency_matrix(similarity_matrix, k)

    # Compute Laplacian matrix
    laplacian_matrix = laplacian_matrix(adj_matrix)

    # Compute eigenvectors of Laplacian matrix
    eigenvalues, eigenvectors = compute_eigenvectors(laplacian_matrix, num_clusters)

    # K-means clustering using eigenvectors
    cluster_labels = kmeans_clustering(eigenvectors, num_clusters)

    # Plot clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b']
    for i in range(cluster_labels.max() + 1):
        ax.scatter(points_gdf.geometry.x[cluster_labels == i], points_gdf.geometry.y[cluster_labels == i], c=colors[i], label=f'Cluster {i+1}')
    ax.legend()
    plt.title('Spectral Clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    return cluster_labels

def km_euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two points for K-Means."""
    return np.sqrt(np.sum((x1 - x2)**2))

def km_initialize_centroids(X, n_clusters, random_state=None):
    """Initialize centroids for K-Means clustering."""
    np.random.seed(random_state)
    centroids_indices = np.random.choice(X.shape[0], size=n_clusters, replace=False)
    centroids = X[centroids_indices]
    return centroids

def km_assign_clusters(X, centroids):
    """Assign clusters to points based on current centroids for K-Means."""
    clusters = []
    for x in X:
        distances = [km_euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def km_update_centroids(X, clusters, n_clusters):
    """Update centroids based on current cluster assignments for K-Means."""
    centroids = np.zeros((n_clusters, X.shape[1]))
    for cluster in range(n_clusters):
        cluster_points = X[clusters == cluster]
        if len(cluster_points) > 0:
            centroids[cluster] = np.mean(cluster_points, axis=0)
    return centroids

def km_plot_clusters(X, clusters, centroids):
    """Plot clusters and centroids for K-Means."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=150, edgecolor='k')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clusters and Centroids (K-Means)')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def km_kmeans_clustering(shp_file_path, n_clusters=3, max_iterations=100, visualize=True, random_state=None):
    """Perform K-Means clustering on point data from an SHP file."""
    # Suppress warnings for demonstration purposes
    warnings.filterwarnings("ignore")

    # Load the SHP file into a GeoDataFrame
    gdf = gpd.read_file(shp_file_path)

    # Extract features (coordinates) from the GeoDataFrame
    X = np.column_stack((gdf['geometry'].x, gdf['geometry'].y))

    # Initialize centroids
    centroids = km_initialize_centroids(X, n_clusters, random_state=random_state)

    # Perform K-Means iterations
    for _ in range(max_iterations):
        # Assign clusters to points
        clusters = km_assign_clusters(X, centroids)
        # Update centroids
        new_centroids = km_update_centroids(X, clusters, n_clusters)
        # Check for convergence
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    if visualize:
        # Visualize the clusters and centroids
        km_plot_clusters(X, clusters, centroids)

    return clusters, centroids

def db_range_query(X, point, eps):
    """Find neighboring points within a specified distance for DBSCAN."""
    neighbors = []
    for i, x in enumerate(X):
        if km_euclidean_distance(point, x) <= eps:
            neighbors.append(i)
    return neighbors

def db_expand_cluster(X, labels, point, neighbors, cluster_label, eps, min_samples):
    """Expand a cluster based on DBSCAN algorithm."""
    labels[point] = cluster_label
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_label
            new_neighbors = db_range_query(X, X[neighbor], eps)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_label
        i += 1

def db_dbscan_clustering(shp_file_path, eps=0.1, min_samples=5, visualize=True):
    """Perform DBSCAN clustering on point data from an SHP file."""
    # Load the SHP file into a GeoDataFrame
    gdf = gpd.read_file(shp_file_path)

    # Extract features (coordinates) from the GeoDataFrame
    X = np.column_stack((gdf['geometry'].x, gdf['geometry'].y))

    # Initialize labels array
    labels = np.zeros(len(X), dtype=int)  # 0 represents unvisited points
    current_label = 0

    for i, point in enumerate(X):
        if labels[i] != 0:
            continue

        neighbors = db_range_query(X, point, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            current_label += 1
            db_expand_cluster(X, labels, i, neighbors, current_label, eps, min_samples)

    if visualize:
        db_visualize_clusters(X, labels)

    return labels

def db_visualize_clusters(X, labels):
    """Visualize clusters from DBSCAN clustering."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - 1  # Exclude noise points

    plt.figure(figsize=(8, 6))

    for cluster_label in range(1, n_clusters + 1):
        cluster_points = X[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

def hierarchical_clustering(points_shapefile, num_clusters):
    """
    Perform hierarchical clustering on a set of points from a shapefile.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    num_clusters (int): Number of clusters to partition the data into.

    Returns:
    list: Cluster labels for each point.
    """

    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)
    points = list(zip(points_gdf.geometry.x, points_gdf.geometry.y))

    # Calculate pairwise distances between points
    n = len(points)
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            distances[i][j] = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5

    # Initialize clusters
    clusters = [[i] for i in range(n)]

    # Merge closest clusters until specified number of clusters is reached
    while len(clusters) > num_clusters:
        min_dist = float('inf')
        merge_i, merge_j = -1, -1

        # Find closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for point_i in clusters[i]:
                    for point_j in clusters[j]:
                        if distances[point_i][point_j] < min_dist:
                            min_dist = distances[point_i][point_j]
                            merge_i, merge_j = i, j

        # Merge the two closest clusters
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    # Assign cluster labels to each point
    cluster_labels = [0] * n
    for i, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[point] = i

    # Plot clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Color palette for clusters
    for i in range(len(clusters)):
        cluster_points = [points[j] for j in clusters[i]]
        ax.scatter([point[0] for point in cluster_points], [point[1] for point in cluster_points], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    ax.legend()
    plt.title('Hierarchical Clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    return cluster_labels

def louvain_euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    p1 (tuple): Coordinates of the first point.
    p2 (tuple): Coordinates of the second point.

    Returns:
    float: Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def louvain_community_detection(points_shapefile, max_iterations=100, tolerance=1e-6):
    """
    Perform community detection on a set of points from a shapefile using the Louvain algorithm.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    max_iterations (int): Maximum number of iterations for the Louvain algorithm.
    tolerance (float): Convergence threshold for the Louvain algorithm.

    Returns:
    list: Community labels for each point.
    """
    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)
    num_points = len(points_gdf)

    # Extract coordinates from GeoDataFrame
    coordinates = np.array(points_gdf.geometry.apply(lambda point: (point.x, point.y)))

    # Create adjacency matrix
    adjacency_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            # Calculate Euclidean distance as edge weight
            dist = louvain_euclidean_distance(coordinates[i], coordinates[j])
            # Assign weight to adjacency matrix
            adjacency_matrix[i, j] = dist
            adjacency_matrix[j, i] = dist

    # Initialize community labels for each point
    community_labels = list(range(num_points))

    # Louvain algorithm
    prev_modularity = -1
    for _ in range(max_iterations):
        # Initialize change flag and modularity
        change = False
        modularity = 0

        # Iterate over each point
        for i in range(num_points):
            # Calculate the modularity change by moving the point to its neighboring communities
            current_community = community_labels[i]
            best_community = current_community
            max_modularity_change = 0
            
            # Find neighboring communities
            neighbors = np.unique([community_labels[j] for j in np.nonzero(adjacency_matrix[i])[0]])
            
            # Calculate modularity change for each neighboring community
            for neighbor in neighbors:
                if neighbor != current_community:
                    # Calculate modularity change
                    modularity_change = 0
                    for j in range(num_points):
                        if community_labels[j] == current_community:
                            modularity_change -= adjacency_matrix[i, j]
                        elif community_labels[j] == neighbor:
                            modularity_change += adjacency_matrix[i, j]
                    
                    # Update best community if modularity change is positive
                    if modularity_change > max_modularity_change:
                        max_modularity_change = modularity_change
                        best_community = neighbor
            
            # Move the point to the community with the maximum modularity change
            if best_community != current_community:
                community_labels[i] = best_community
                change = True
        
        # Calculate modularity
        for i in range(num_points):
            for j in range(num_points):
                if community_labels[i] == community_labels[j]:
                    modularity += adjacency_matrix[i, j]

        # Check for convergence
        if abs(modularity - prev_modularity) < tolerance:
            break
        prev_modularity = modularity

        # If no change occurred, stop iterating
        if not change:
            break

    # Plot communities
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Color palette for communities
    for i, community in enumerate(np.unique(community_labels)):
        community_points = points_gdf.geometry[np.array(community_labels) == community]
        ax.scatter([point.x for point in community_points], [point.y for point in community_points], c=colors[i % len(colors)], label=f'Community {i+1}')
    ax.legend()
    plt.title('Community Detection using Louvain Algorithm')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    return community_labels

def op_euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    p1 (tuple): Coordinates of the first point.
    p2 (tuple): Coordinates of the second point.

    Returns:
    float: Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def op_optics_cluster(points_gdf, epsilon, min_pts):
    """
    Perform clustering on a set of points using the OPTICS algorithm.

    Args:
    points_gdf (geopandas.GeoDataFrame): GeoDataFrame containing point data.
    epsilon (float): Maximum distance to consider points as neighbors.
    min_pts (int): Minimum number of points required to form a cluster.

    Returns:
    list: Cluster labels for each point.
    """
    num_points = len(points_gdf)
    # Extract coordinates from GeoDataFrame
    coordinates = np.array(points_gdf.geometry.apply(lambda point: (point.x, point.y)))
    # Initialize reachability distances and cluster labels
    reachability_distances = np.full(num_points, np.inf)
    cluster_labels = np.full(num_points, -1)
    # Iterate over each point
    for i in range(num_points):
        # Calculate distances to all other points
        distances = [op_euclidean_distance(coordinates[i], coordinates[j]) for j in range(num_points)]
        # Find neighbors within epsilon distance
        neighbors = [j for j in range(num_points) if distances[j] <= epsilon and j != i]
        if len(neighbors) >= min_pts:
            # If core point, update reachability distances of neighbors
            for neighbor in neighbors:
                core_dist = max(distances[j] for j in neighbors if j != neighbor)
                reachability_distances[neighbor] = min(core_dist, distances[neighbor])
        # Assign point to a cluster if not already assigned
        if cluster_labels[i] == -1:
            if len(neighbors) >= min_pts:
                # If core point, expand cluster
                cluster_id = i
                while True:
                    cluster_labels[cluster_id] = cluster_id
                    neighbors = [j for j in range(num_points) if distances[j] <= epsilon and j != cluster_id]
                    if len(neighbors) == 0:
                        break
                    # Find next core point with minimum reachability distance
                    next_core = min(neighbors, key=lambda x: reachability_distances[x])
                    core_dist = max(distances[j] for j in neighbors if j != next_core)
                    if reachability_distances[next_core] > core_dist:
                        # If next core point is not directly reachable, mark it as outlier
                        cluster_labels[next_core] = -1
                    else:
                        # If next core point is directly reachable, expand cluster
                        cluster_id += 1
                        cluster_labels[next_core] = cluster_id
                        cluster_id = next_core
            else:
                # If outlier point, mark as noise
                cluster_labels[i] = -1
    return cluster_labels

def op_plot_clusters(points_gdf, cluster_labels):
    """
    Plot clusters on a map.

    Args:
    points_gdf (geopandas.GeoDataFrame): GeoDataFrame containing point data.
    cluster_labels (list): Cluster labels for each point.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Color palette for clusters
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            # Plot outliers as black
            outlier_points = points_gdf.geometry[cluster_labels == -1]
            ax.scatter([point.x for point in outlier_points], [point.y for point in outlier_points], c='k', label='Outliers')
        else:
            # Plot cluster points with different colors
            cluster_points = points_gdf.geometry[cluster_labels == cluster_id]
            ax.scatter([point.x for point in cluster_points], [point.y for point in cluster_points], c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}')
    ax.legend()
    plt.title('Clustering using OPTICS Algorithm')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def op_optics_clustering(points_shapefile, epsilon, min_pts):
    """
    Perform clustering on a set of points from a shapefile using the OPTICS algorithm and plot the clusters.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    epsilon (float): Maximum distance to consider points as neighbors.
    min_pts (int): Minimum number of points required to form a cluster.
    """
    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)
    # Perform OPTICS clustering
    cluster_labels = op_optics_cluster(points_gdf, epsilon, min_pts)
    # Plot clusters
    op_plot_clusters(points_gdf, cluster_labels)

def GMM_gaussian(x, mean, cov):
    """
    Calculate the value of the Gaussian distribution at point x.

    Args:
    x (np.array): The point at which to evaluate the Gaussian.
    mean (np.array): The mean of the Gaussian distribution.
    cov (np.array): The covariance matrix of the Gaussian distribution.

    Returns:
    float: The value of the Gaussian distribution at point x.
    """
    n = len(x)
    coef = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5)
    exponent = -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
    return coef * np.exp(exponent)

def GMM_initialize_parameters(data, num_clusters):
    """
    Initialize the parameters of the GMM.

    Args:
    data (np.array): The data points.
    num_clusters (int): The number of clusters.

    Returns:
    tuple: Tuple containing means, covariances, and priors for each cluster.
    """
    n, m = data.shape
    # Randomly select data points as initial means
    means = data[np.random.choice(n, num_clusters, replace=False)]
    # Use empirical covariance matrix as initial covariance for each cluster
    covariances = [np.cov(data.T) for _ in range(num_clusters)]
    # Use uniform prior probability
    priors = [1 / num_clusters] * num_clusters
    return means, covariances, priors
    

def GMM_expectation(data, means, covariances, priors):
    """
    Perform the expectation step of the EM algorithm.

    Args:
    data (np.array): The data points.
    means (list): List of mean vectors for each cluster.
    covariances (list): List of covariance matrices for each cluster.
    priors (list): List of prior probabilities for each cluster.

    Returns:
    np.array: Array of responsibilities for each data point and cluster.
    """
    num_clusters = len(means)
    n = len(data)
    responsibilities = np.zeros((n, num_clusters))
    for i in range(n):
        for j in range(num_clusters):
            responsibilities[i, j] = priors[j] * GMM_gaussian(data[i], means[j], covariances[j])
        responsibilities[i, :] /= np.sum(responsibilities[i, :])
    return responsibilities

def GMM_maximization(data, responsibilities):
    """
    Perform the maximization step of the EM algorithm.

    Args:
    data (np.array): The data points.
    responsibilities (np.array): Array of responsibilities for each data point and cluster.

    Returns:
    tuple: Tuple containing updated means, covariances, and priors for each cluster.
    """
    n, m = data.shape
    num_clusters = responsibilities.shape[1]
    means = np.zeros((num_clusters, m))
    covariances = [np.zeros((m, m)) for _ in range(num_clusters)]
    priors = np.zeros(num_clusters)

    for j in range(num_clusters):
        for i in range(n):
            means[j] += responsibilities[i, j] * data[i]
            covariances[j] += responsibilities[i, j] * np.outer(data[i] - means[j], data[i] - means[j])
            priors[j] += responsibilities[i, j]
        means[j] /= np.sum(responsibilities[:, j])
        covariances[j] /= np.sum(responsibilities[:, j])
        priors[j] /= n

    return means, covariances, priors

def GMM_compute_log_likelihood(data, means, covariances, priors):
    """
    Compute the log likelihood of the data under the GMM model.

    Args:
    data (np.array): The data points.
    means (list): List of mean vectors for each cluster.
    covariances (list): List of covariance matrices for each cluster.
    priors (list): List of prior probabilities for each cluster.

    Returns:
    float: The log likelihood of the data under the GMM model.
    """
    log_likelihood = 0
    for x in data:
        likelihood = sum([prior * GMM_gaussian(x, mean, cov) for mean, cov, prior in zip(means, covariances, priors)])
        log_likelihood += np.log(likelihood)
    return log_likelihood

def GMM_clustering(points_shapefile, num_clusters, max_iterations=100, tolerance=1e-6):
    """
    Perform GMM clustering on a set of points from a shapefile.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    num_clusters (int): Number of clusters to partition the data into.
    max_iterations (int): Maximum number of iterations for the EM algorithm.
    tolerance (float): Convergence threshold for the EM algorithm.

    Returns:
    np.array: Cluster labels for each point.
    """
    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)
    data = np.array([[point.x, point.y] for point in points_gdf.geometry])

    # Initialize parameters
    means, covariances, priors = GMM_initialize_parameters(data, num_clusters)

    # EM algorithm
    prev_log_likelihood = float('-inf')
    for iteration in range(max_iterations):
        # Expectation step
        responsibilities = GMM_expectation(data, means, covariances, priors)

        # Maximization step
        means, covariances, priors = GMM_maximization(data, responsibilities)

        # Compute log likelihood
        log_likelihood = GMM_compute_log_likelihood(data, means, covariances, priors)

        # Check for convergence
        if log_likelihood - prev_log_likelihood < tolerance:
            break
        prev_log_likelihood = log_likelihood

    # Assign cluster labels
    cluster_labels = np.argmax(responsibilities, axis=1)

    # Plot clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Color palette for clusters
    for i in range(num_clusters):
        cluster_points = data[cluster_labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    ax.legend()
    plt.title('GMM Clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    return cluster_labels

def hdbscan_clustering(points_shapefile, min_samples, min_cluster_size):
    """
    Perform clustering on a set of points from a shapefile using the HDBSCAN algorithm and plot the clusters.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
    min_cluster_size (int): The minimum number of points required to form a cluster.
    """
    # Define a function to calculate Euclidean distance between two points
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Define a function to expand a cluster starting from a core point
    def expand_cluster(i, distances, cluster_labels, cluster_id, min_cluster_size):
        cluster_labels[i] = cluster_id
        seeds = [j for j in range(len(cluster_labels)) if distances[i, j] < min_samples]
        while seeds:
            j = seeds.pop(0)
            if cluster_labels[j] == -1:
                cluster_labels[j] = cluster_id
                if np.sum(distances[j] < min_samples) >= min_samples:
                    seeds.extend([k for k in range(len(cluster_labels)) if distances[j, k] < min_samples])
            elif cluster_labels[j] == 0:
                cluster_labels[j] = cluster_id

    # Define a function to plot clusters on a map
    def plot_clusters(points_gdf, cluster_labels):
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:
                outlier_points = points_gdf.geometry[cluster_labels == -1]
                ax.scatter([point.x for point in outlier_points], [point.y for point in outlier_points], c='k', label='Outliers')
            else:
                cluster_points = points_gdf.geometry[cluster_labels == cluster_id]
                ax.scatter([point.x for point in cluster_points], [point.y for point in cluster_points], c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}')
        ax.legend()
        plt.title('Clustering using HDBSCAN Algorithm')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    # Load shapefile point data
    points_gdf = gpd.read_file(points_shapefile)
    num_points = len(points_gdf)
    coordinates = np.array(points_gdf.geometry.apply(lambda point: (point.x, point.y)))

    # Compute pairwise distances
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            distances[i, j] = euclidean_distance(coordinates[i], coordinates[j])
            distances[j, i] = distances[i, j]

    # Compute core points
    core_points = np.sum(distances < min_samples, axis=1) >= min_samples

    # Initialize cluster labels
    cluster_labels = np.full(num_points, -1)
    cluster_id = 0

    # Iterate over each point
    for i in range(num_points):
        if cluster_labels[i] == -1 and core_points[i]:
            cluster_id += 1
            expand_cluster(i, distances, cluster_labels, cluster_id, min_cluster_size)

    # Plot clusters
    plot_clusters(points_gdf, cluster_labels)




# Example usage:
if __name__ == "__main__":
    density_data_path = './output/AKDE_density_grid.npy'
    hotspots = extract_hotspots(density_data_path)
    visualize_hotspots(np.load(density_data_path), hotspots)
