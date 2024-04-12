# API Reference

The API Reference provides an overview of all public objects, functions and methods implemented in GeoHD.

## Process

### `calculate_kernel_density(points_gdf, bandwidth)`

Calculates the kernel density estimation (KDE) from a GeoDataFrame containing point geometries.

#### Parameters:
- `points_gdf` (*geopandas.GeoDataFrame*): A GeoDataFrame that includes point geometries representing the locations of interest.
- `bandwidth` (*float*): The bandwidth parameter for the KDE, which controls the smoothness of the resulting density estimate.

#### Returns:
- `gaussian_kde` (*scipy.stats.gaussian_kde*): An object representing the KDE computed from the point coordinates.

#### Details:
This function extracts the x and y coordinates from the geometries in the GeoDataFrame and uses these coordinates to compute the KDE. The bandwidth parameter is used as the method for smoothing and is passed to the `gaussian_kde` function from `scipy.stats`.

### `generate_density_raster(kde, xmin, ymin, xmax, ymax, pixel_size)`

Generates a density raster from the KDE object.

#### Parameters:
- `kde` (*scipy.stats.gaussian_kde*): The KDE object obtained from the `calculate_kernel_density` function.
- `xmin`, `ymin`, `xmax`, `ymax` (*float*): The bounding box coordinates that define the extent of the raster.
- `pixel_size` (*float*): The size of each pixel in the raster.

#### Returns:
- `density_raster` (*numpy.ndarray*): A 2D NumPy array representing the density raster.

#### Details:
This function creates a grid covering the specified bounding box using the provided pixel size. It then evaluates the KDE at each grid point and returns a raster array where each cell contains the estimated density value.

### `process_shapefile(input_file_path, bandwidth=0.2, pixel_size=0.001)`

Processes a shapefile to generate a KDE raster and applies a Gaussian filter for smoothing.

#### Parameters:
- `input_file_path` (*str*): The file path to the input shapefile.
- `bandwidth` (*float*): The bandwidth for KDE. Default is 0.2.
- `pixel_size` (*float*): The pixel size for the density raster. Default is 0.001.

#### Returns:
- `smoothed_density` (*numpy.ndarray*): A smoothed density raster array.

#### Details:
This function reads the shapefile, calculates the KDE, generates a raster, and applies a Gaussian filter for smoothing. The Gaussian filter uses a sigma value of 1.5 to reduce noise in the density raster.

### `plot_density_raster(smoothed_density, output_data_path, xmin, ymin, xmax, ymax)`

Plots and saves the density raster.

#### Parameters:
- `smoothed_density` (*numpy.ndarray*): The smoothed density raster array.
- `output_data_path` (*str*): The file path to save the density data as a NumPy array.
- `xmin`, `ymin`, `xmax`, `ymax` (*float*): The bounding box coordinates for plotting the raster.

#### Details:
This function creates a plot of the smoothed density raster using Matplotlib. It uses the 'Purples' colormap and displays a colorbar with the label 'Density'. The density data is also saved as a NumPy array at the specified output path.

### `kernel_density_estimation_from_shapefile(shapefile_path, bandwidth=0.1, grid_resolution=100)`

Estimates the kernel density directly from a shapefile and plots the resulting density map.

#### Parameters:
- `shapefile_path` (*str*): The file path to the shapefile containing the point data.
- `bandwidth` (*float*): The bandwidth for KDE. Default is 0.1.
- `grid_resolution` (*int*): The number of points along each axis of the grid. Default is 100.

#### Details:
This function reads the shapefile and extracts the x and y coordinates of the points to create a grid. It then computes the KDE by calculating the distance from each point to every grid point and applying a Gaussian kernel. The resulting density grid is plotted with the original points overlayed.

### `adaptiveKDE(shp_file_path, output_data_path)`

Performs Adaptive Kernel Density Estimation on a set of points from a given shapefile and saves the density grid.
KDE is a widely used technique in spatial analysis to smooth and visualize the distribution of spatial points.The bandwidth parameter is crucial in KDE as it controls the trade-off between bias and variance in the estimation. A smaller bandwidth leads to a more varied and detailed estimate (low bias, high variance), while a larger bandwidth results in a smoother estimate (high bias, low variance).
    
The calculation of KDE involves the following steps:
1. Extract the coordinates from the GeoDataFrame and form a two-dimensional array.
2. Apply the Gaussian kernel function to each point, which is centered at the point and has a spread determined by the bandwidth.
3. Sum the contributions from all points to obtain the estimated density at any location.
4. Normalize the resulting density estimate so that it integrates to 1 over the entire space.
    

#### Parameters:
- `shp_file_path` (*str*): The file path to the input shapefile containing the point data.
- `output_data_path` (*str*): The file path where the density data will be saved.

#### Returns:
- `None`: This function does not return a value but saves the density grid to the specified output path.

#### Details:
The `adaptiveKDE` function reads the shapefile, extracts the point geometries, and calculates the KDE using an adaptive bandwidth selection method. The function iteratively refines the bandwidth parameter (`h`) and the power parameter (`a`) to achieve an optimal balance between bias and variance in the KDE. The final density grid is saved as a NumPy array at the specified output path.

### `plot_density_grid(points, h, output_data_path, resolution=1000)`

Generates a density estimation grid and visualizes it as a contour plot.

#### Parameters:
- `points` (*dict*): A dictionary containing the point data.
- `h` (*float*): The bandwidth parameter for KDE.
- `output_data_path` (*str*): The file path where the density data will be saved.
- `resolution` (*int*): The grid resolution for plotting. Default is 1000.

#### Returns:
- `None`: This function does not return a value but saves the density data and displays a contour plot.

#### Details:
The `plot_density_grid` function creates a grid covering the spatial extent of the input points and computes the KDE at each grid cell. It then saves the density values as a NumPy array and visualizes the density distribution with a contour plot using Matplotlib.

## Analysis

### `plot_g_function(data_path)`

Plots the G Function, which is a measure of the spatial distribution of points, against distance.

#### Parameters:
- `data_path` (*str*): The file path to the shapefile (.shp) containing spatial point data.

#### Returns:
- `None`: This function does not return a value but generates a plot showing the G Function.

#### Details:
The `plot_g_function` function reads spatial point data from a shapefile and computes the G Function. It plots the observed G Function statistic along with simulated values to provide a comparison. Points with p-values less than 0.01 are highlighted on the plot.

### `plot_f_function(data_path)`

Plots the F Function, which is a measure of the spatial distribution of points within a given distance of each point.

#### Parameters:
- `data_path` (*str*): The file path to the shapefile (.shp) containing spatial point data.

#### Returns:
- `None`: This function does not return a value but generates a plot showing the F Function.

#### Details:
The `plot_f_function` function reads spatial point data from a shapefile and computes the F Function. It plots the observed F Function statistic along with a matrix of simulated values. Points with p-values less than 0.05 are highlighted on the plot.

### `plot_j_function(data_path)`

Plots the J Function, which is a cumulative version of the G Function, against distance.

#### Parameters:
- `data_path` (*str*): The file path to the shapefile (.shp) containing spatial point data.

#### Returns:
- `None`: This function does not return a value but generates a plot showing the J Function.

#### Details:
The `plot_j_function` function reads spatial point data from a shapefile and computes the J Function. It plots the observed J Function statistic and compares it to the expected value (horizontal line at 1).

### `plot_k_function(data_path)`

Plots the K Function, which is a measure of the spatial distribution of points and compares it to a complete spatial randomness (CSR) model.

#### Parameters:
- `data_path` (*str*): The file path to the shapefile (.shp) containing spatial point data.

#### Returns:
- `None`: This function does not return a value but generates a plot showing the K Function.

#### Details:
The `plot_k_function` function reads spatial point data from a shapefile and computes the K Function. It plots the observed K Function statistic along with simulated values. Points with p-values less than 0.05 are highlighted on the plot.

### `plot_l_function(data_path)`

Plots the L Function, which is the cumulative distribution function of the K Function, against distance.

#### Parameters:
- `data_path` (*str*): The file path to the shapefile (.shp) containing spatial point data.

#### Returns:
- `None`: This function does not return a value but generates a plot showing the L Function.

#### Details:
The `plot_l_function` function reads spatial point data from a shapefile and computes the L Function. It plots the observed L Function statistic along with simulated values. Points with p-values less than 0.05 are highlighted on the plot.

### `create_cell_zones(area_file, crash_file, num_rows=30, num_cols=30)`

This function creates a grid of rectangular cells over a specified study area, identifies which cells intersect with the study area polygons, and overlays these cells with crash points to provide a detailed spatial analysis.

#### Parameters:
- `area_file` (*str*): The file path to the shapefile containing the study area polygons.
- `crash_file` (*str*): The file path to the shapefile containing the crash points.
- `num_rows` (*int*, optional): The number of rows for the grid. Default is 30.
- `num_cols` (*int*, optional): The number of columns for the grid. Default is 30.

#### Returns:
- `None`: The function does not return a value but generates a visualization of the intersecting grid cells and crash points.

#### Detailed Description:
`create_cell_zones` begins by reading the polygons that define the study area and the points representing crash locations. It ensures that both datasets are in the same CRS for accurate spatial analysis. The function then calculates the bounding box of the study area and divides it into a grid based on the specified number of rows and columns.
Each grid cell is created as a polygon and added to a GeoDataFrame. The function then identifies which grid cells intersect with the study area polygons using a spatial join operation. The resulting GeoDataFrame is visualized with the study area polygons in light blue, intersecting grid cells in red, and crash points in blue.

### `create_hex_grid_zones(area_file, crash_file, resolution=8)`

This function generates a hexagonal grid over the study area, identifies intersecting hexagons, and overlays this grid with crash points for a comprehensive spatial analysis.

#### Parameters:
- `area_file` (*str*): The file path to the shapefile containing the study area polygons.
- `crash_file` (*str*): The file path to the shapefile containing the crash points.
- `resolution` (*int*, optional): The resolution of the H3 hexagons. Higher resolution values result in smaller hexagons. Default is 8.

#### Returns:
- `None`: The function does not return a value but generates a visualization of the intersecting hexagon grid cells and crash points.

#### Detailed Description:
`create_hex_grid_zones` reads the study area and crash point datasets, ensuring they share the same CRS. It then calculates the bounding box of the study area and converts it into H3 hexagons at the specified resolution. Each hexagon is converted into a polygon and added to a GeoDataFrame.
The function performs a spatial join between the hexagons and the study area polygons to identify intersecting hexagons. The resulting GeoDataFrame is visualized, with the study area polygons in light blue, intersecting hexagon grid cells in red, and crash points in blue.

### `create_cell_heatmap(area_file, crash_file, num_rows=30, num_cols=30)`

This function generates a heatmap of crash point density within the study area by dividing it into a grid of rectangular cells and counting the number of crash points within each cell.

#### Parameters:
- `area_file` (*str*): The file path to the shapefile containing the study area polygons.
- `crash_file` (*str*): The file path to the shapefile containing the crash points.
- `num_rows` (*int*, optional): The number of rows for the grid. Default is 30.
- `num_cols` (*int*, optional): The number of columns for the grid. Default is 30.

#### Returns:
- `None`: The function does not return a value but generates a heatmap of crash point density.

#### Detailed Description:
`create_cell_heatmap` starts by creating a grid of polygons over the study area, similar to `create_cell_zones`. It then performs a spatial join between the crash points and the grid cells to identify which points fall within each cell. The number of crash points within each cell is counted and stored in a new GeoDataFrame.
The function calculates the density of crash points by dividing the count by the area of each cell. This density is then visualized as a heatmap, with colors ranging from light blue (low density) to dark blue (high density). The heatmap is overlaid on the study area, with the study area polygons and crash points also displayed for context.

### `create_hexagonal_heatmap(area_file, crash_file, resolution=8)`

This function generates a heatmap of crash point density within the study area using a hexagonal grid, providing a more efficient and visually appealing representation of spatial data.

#### Parameters:
- `area_file` (*str*): The file path to the shapefile containing the study area polygons.
- `crash_file` (*str*): The file path to the shapefile containing the crash points.
- `resolution` (*int*, optional): The resolution of the H3 hexagons. Higher resolution values result in smaller hexagons. Default is 8.

#### Returns:
- `None`: The function does not return a value but generates a heatmap of crash point density.

#### Detailed Description:
`create_hexagonal_heatmap` operates similarly to `create_cell_heatmap`, but it uses hexagonal grids instead of rectangular cells. The function converts the bounding box of the study area into H3 hexagons at the specified resolution and counts the number of crash points within each hexagon.
The density of crash points is calculated and visualized as a heatmap. The hexagonal shape of the grid cells allows for a more efficient coverage of the area and can provide a more visually appealing and easily interpretable heatmap. The heatmap is color-coded based on the density of crash points, with the study area polygons and crash points also displayed for reference.

## Visualization

### `visualize_shapefile(file_path, output_image_path='./output/point_data.png')`

This function creates a visualization of the spatial data contained within a shapefile and overlays it on a map background sourced from `contextily`.

#### Parameters:
- `file_path` (*str*): A string representing the path to the shapefile (.shp) to be visualized.
- `output_image_path` (*str*, optional): A string representing the path where the visualization image should be saved. Default is `'./output/point_data.png'`.

#### Raises:
- `FileNotFoundError`: If the specified shapefile does not exist or is not a valid shapefile.
- `ValueError`: If the `output_image_path` is not provided or is invalid.

#### Returns:
- `None`: The function does not return a value but generates a visualization and saves it as an image file.

#### Detailed Description:
The `visualize_shapefile` function is a powerful tool for quickly understanding the spatial distribution of data within a shapefile. It first checks the validity of the shapefile using the `is_valid_shapefile` function. If the shapefile is valid, it reads the data into a GeoDataFrame and prints a summary of the first few rows for reference.
The function then creates a plot using `matplotlib`, where the spatial data is plotted as points with customizable markers. To provide context to the data, a map background is added using `contextily`, which integrates seamlessly with `matplotlib`. The map background is selected based on the Coordinate Reference System (CRS) of the GeoDataFrame, ensuring accurate placement of the data on the map.
The visualization includes setting the axis limits to encompass the entire spatial extent of the data, adding a title, and displaying the plot. After displaying the plot, the function saves the visualization as an image file at the specified `output_image_path`. This saved image can be used for presentations, reports, or further analysis.

## Utils

### `is_valid_shapefile(file_path)`

This function checks whether a given file path corresponds to a valid shapefile by attempting to read it using `geopandas`.

#### Parameters:
- `file_path` (*str*): A string representing the path to the shapefile (.shp) to be validated.

#### Returns:
- `bool`: Returns `True` if the shapefile is valid and can be read without errors; `False` otherwise.

#### Detailed Description:
The `is_valid_shapefile` function is designed to provide a quick and reliable method for determining the integrity of a shapefile. It is useful for ensuring that the data is in the correct format before proceeding with further analysis or visualization. The function encapsulates the file reading process within a try-except block, catching exceptions related to file not found or missing mandatory files, which are common issues when dealing with shapefiles.

### `extract_hotspots(density_data_path, window_size=10, threshold=0)`

This function extracts hotspots from density data using a window analysis approach, which is particularly useful for identifying areas of concentrated activity or points of interest within a given spatial dataset.

#### Parameters:
- `density_data_path` (*str*): The file path to the .npy file containing the density data.
- `window_size` (*int*, optional): The size of the window for the maximum filter. This parameter determines the extent of the local area considered when identifying hotspots. Default is 10.
- `threshold` (*int*, optional): The threshold value used to classify regions as extreme (hotspots) or non-extreme. Default is 0.

#### Returns:
- `hotspots` (*numpy.ndarray*): An array containing the coordinates of the identified hotspots.

#### Detailed Description:
The `extract_hotspots` function begins by loading the density data from a .npy file. It then applies a maximum filter within a defined window size to identify local maxima. By subtracting the original density data from the maximum filter result, a non-negative surface is obtained, which helps to pinpoint areas of high density relative to their local surroundings. The function classifies regions as extreme if their density equals or exceeds the specified threshold. Finally, it extracts the coordinates of these extreme regions, which are considered hotspots.

### `visualize_hotspots(density_data, hotspots)`

This function visualizes the density data and the identified hotspots, providing a clear and intuitive representation of the spatial distribution of high-density areas.

#### Parameters:
- `density_data` (*numpy.ndarray*): A 2D array containing the density data.
- `hotspots` (*numpy.ndarray*): An array containing the coordinates of the hotspots as returned by the `extract_hotspots` function.

#### Detailed Description:
The `visualize_hotspots` function uses `imshow` from `matplotlib` to display the density data as a heatmap, with the color intensity representing the density levels. It adds a colorbar to the plot for reference. The hotspots are then plotted as individual red points on top of the density map, allowing for easy identification of the locations with the highest density values. The function sets appropriate labels and titles for the axes and the plot, and includes a legend to distinguish between the density data and the hotspots.

### `km_euclidean_distance(x1, x2)`
Calculates the Euclidean distance between two points, which is used in the K-Means clustering algorithm.

#### Parameters:
- `x1` (*numpy.ndarray*): A vector representing the coordinates of the first point.
- `x2` (*numpy.ndarray*): A vector representing the coordinates of the second point.

#### Returns:
- *float*: The Euclidean distance between the two points.

### `km_initialize_centroids(X, n_clusters, random_state=None)`
Initializes the centroids for the K-Means clustering algorithm.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `n_clusters` (*int*): The number of clusters to form.
- `random_state` (*int*, optional): The seed for the random number generator. If `None`, the current system time is used.

#### Returns:
- *numpy.ndarray*: An array containing the coordinates of the initialized centroids.

### `km_assign_clusters(X, centroids)`
Assigns clusters to points based on the current centroids for K-Means clustering.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `centroids` (*numpy.ndarray*): A 2D array containing the coordinates of the centroids.

#### Returns:
- *numpy.ndarray*: An array of cluster assignments for each point.

### `km_update_centroids(X, clusters, n_clusters)`
Updates the centroids based on the current cluster assignments for K-Means clustering.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `clusters` (*numpy.ndarray*): An array of cluster assignments for each point.
- `n_clusters` (*int*): The number of clusters.

#### Returns:
- *numpy.ndarray*: An array containing the updated coordinates of the centroids.

### `km_plot_clusters(X, clusters, centroids)`
Plots the clusters and centroids for K-Means clustering.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `clusters` (*numpy.ndarray*): An array of cluster assignments for each point.
- `centroids` (*numpy.ndarray*): A 2D array containing the coordinates of the centroids.

### `km_kmeans_clustering(shp_file_path, n_clusters=3, max_iterations=100, visualize=True, random_state=None)`
Performs K-Means clustering on point data from an SHP file and visualizes the results.

#### Parameters:
- `shp_file_path` (*str*): The file path to the SHP file containing the point data.
- `n_clusters` (*int*, optional): The number of clusters to form. Default is 3.
- `max_iterations` (*int*, optional): The maximum number of iterations for the K-Means algorithm. Default is 100.
- `visualize` (*bool*, optional): Whether to visualize the clusters and centroids. Default is True.
- `random_state` (*int*, optional): The seed for the random number generator. If `None`, the current system time is used.

#### Returns:
- *tuple*: A tuple containing the cluster assignments for each point and the updated centroids.

### `db_range_query(X, point, eps)`
Finds neighboring points within a specified distance for DBSCAN clustering.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `point` (*numpy.ndarray*): A vector representing the coordinates of the point.
- `eps` (*float*): The maximum distance between two samples for one to be considered as in the neighborhood of the other.

#### Returns:
- *list*: A list of indices of neighboring points.

### `db_expand_cluster(X, labels, point, neighbors, cluster_label, eps, min_samples)`
Expands a cluster based on the DBSCAN algorithm.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `labels` (*numpy.ndarray*): An array of labels for each point.
- `point` (*int*): The index of the point.
- `neighbors` (*list*): A list of indices of neighboring points.
- `cluster_label` (*int*): The label for the current cluster.
- `eps` (*float*): The maximum distance between two samples.
- `min_samples` (*int*): The minimum number of samples within a neighborhood specified by `eps` to form a dense region.

### `db_dbscan_clustering(shp_file_path, eps=0.1, min_samples=5, visualize=True)`
Performs DBSCAN clustering on point data from an SHP file and visualizes the results.

#### Parameters:
- `shp_file_path` (*str*): The file path to the SHP file containing the point data.
- `eps` (*float*, optional): The maximum distance between two samples. Default is 0.1.
- `min_samples` (*int*, optional): The minimum number of samples within a neighborhood specified by `eps`. Default is 5.
- `visualize` (*bool*, optional): Whether to visualize the clusters. Default is True.

#### Returns:
- *numpy.ndarray*: An array of labels for each point, where unvisited points are labeled 0, noise points are labeled -1, and cluster labels start from 1.

### `db_visualize_clusters(X, labels)`
Visualizes clusters from DBSCAN clustering.

#### Parameters:
- `X` (*numpy.ndarray*): A 2D array containing the points to be clustered.
- `labels` (*numpy.ndarray*): An array of labels for each point.

### `GMM_gaussian(x, mean, cov)`
Calculates the value of the Gaussian distribution at a given point.

#### Parameters:
- `x` (*numpy.ndarray*): The point at which to evaluate the Gaussian distribution.
- `mean` (*numpy.ndarray*): The mean vector of the Gaussian distribution.
- `cov` (*numpy.ndarray*): The covariance matrix of the Gaussian distribution.

#### Returns:
- *float*: The calculated value of the Gaussian distribution at point x.

### `GMM_initialize_parameters(data, num_clusters)`
Initializes the parameters of the GMM, including the means, covariances, and priors for each cluster.

#### Parameters:
- `data` (*numpy.ndarray*): The data points as a 2D array.
- `num_clusters` (*int*): The number of clusters to initialize.

#### Returns:
- *tuple*: A tuple containing initialized means, covariances, and priors for each cluster.

### `GMM_expectation(data, means, covariances, priors)`
Performs the expectation step of the EM algorithm, calculating the responsibilities (posteriors) for each data point and cluster.

#### Parameters:
- `data` (*numpy.ndarray*): The data points as a 2D array.
- `means` (*list*): List of mean vectors for each cluster.
- `covariances` (*list*): List of covariance matrices for each cluster.
- `priors` (*list*): List of prior probabilities for each cluster.

#### Returns:
- *numpy.ndarray*: A 2D array of responsibilities for each data point and cluster.

### `GMM_maximization(data, responsibilities)`
Performs the maximization step of the EM algorithm, updating the means, covariances, and priors for each cluster.

#### Parameters:
- `data` (*numpy.ndarray*): The data points as a 2D array.
- `responsibilities` (*numpy.ndarray*): A 2D array of responsibilities for each data point and cluster.

#### Returns:
- *tuple*: A tuple containing updated means, covariances, and priors for each cluster.

### `GMM_compute_log_likelihood(data, means, covariances, priors)`
Computes the log likelihood of the data under the GMM model.

#### Parameters:
- `data` (*numpy.ndarray*): The data points as a 2D array.
- `means` (*list*): List of mean vectors for each cluster.
- `covariances` (*list*): List of covariance matrices for each cluster.
- `priors` (*list*): List of prior probabilities for each cluster.

#### Returns:
- *float*: The log likelihood of the data under the GMM model.

### `GMM_clustering(points_shapefile, num_clusters, max_iterations=100, tolerance=1e-6)`
Performs GMM clustering on a set of points from a shapefile and visualizes the resulting clusters.

#### Parameters:
- `points_shapefile` (*str*): Path to the shapefile containing point data.
- `num_clusters` (*int*): Number of clusters to partition the data into.
- `max_iterations` (*int*, optional): Maximum number of iterations for the EM algorithm. Default is 100.
- `tolerance` (*float*, optional): Convergence threshold for the EM algorithm. Default is 1e-6.

#### Returns:
- *numpy.ndarray*: Cluster labels for each point.
