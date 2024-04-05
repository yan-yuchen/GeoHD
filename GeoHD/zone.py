import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import h3


def create_cell_zones(area_file, crash_file, num_rows=30, num_cols=30):
    """
    This function creates a grid of polygons over a study area, identifies the grid cells intersecting with
    provided area polygons, and visualizes them along with crash points within the study area.

    Parameters:
        area_file (str): File path to the shapefile containing the study area polygons.
        crash_file (str): File path to the shapefile containing crash points.
        num_rows (int): Number of rows for the grid.
        num_cols (int): Number of columns for the grid.
    """

    # Read area polygons and crash points data
    area = gpd.read_file(area_file)
    crash_points = gpd.read_file(crash_file)

    # Ensure all geometry objects have the same CRS
    crash_points = crash_points.to_crs(area.crs)

    # Define bounding box of the study area
    xmin, ymin, xmax, ymax = area.total_bounds

    # Calculate grid cell dimensions
    grid_width = (xmax - xmin) / num_cols
    grid_height = (ymax - ymin) / num_rows

    # Create grid polygons
    grid_polygons = []
    for i in range(num_rows):
        for j in range(num_cols):
            x1 = xmin + j * grid_width
            y1 = ymin + i * grid_height
            x2 = x1 + grid_width
            y2 = y1 + grid_height
            grid_polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    # Convert grid polygons to GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons})

    # Add index to grid cells
    grid_gdf['grid_id'] = range(1, len(grid_gdf) + 1)

    # Find grid cells intersecting with area polygons
    intersections = gpd.sjoin(grid_gdf, area, op='intersects')

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    area.plot(ax=ax, color='lightblue', edgecolor='black')  # Plotting study area polygons
    intersections.plot(ax=ax, facecolor='none', edgecolor='red')  # Plotting intersecting grid cells
    crash_points.plot(ax=ax, color='blue', markersize=5)  # Plotting crash points
    plt.title('Intersecting Grids and Crash Points within Study Area')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()



def create_hex_grid_zones(area_file, crash_file, resolution=8):
    """
    This function creates a hexagonal grid over a study area, identifies the grid cells intersecting with
    provided area polygons, and visualizes them along with crash points within the study area.

    Parameters:
        area_file (str): File path to the shapefile containing the study area polygons.
        crash_file (str): File path to the shapefile containing crash points.
        resolution (int): Resolution of the H3 hexagons.
    """


    # Read area polygons and crash points data
    area = gpd.read_file(area_file)
    crash_points = gpd.read_file(crash_file)

    # Ensure all geometry objects have the same CRS
    crash_points = crash_points.to_crs(area.crs)

    # Get bounding box of the study area
    minx, miny, maxx, maxy = area.total_bounds

    # Convert bounding box to H3 hexagons
    hexagons = h3.polyfill_geojson(
        {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]},
        resolution
    )

    # Convert H3 hexagons to polygons
    hex_polygons = [Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True)) for hex_id in hexagons]

    # Convert polygons to GeoDataFrame
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons)

    # Spatial join hexagon grid with original area data
    intersections = gpd.sjoin(hex_gdf, area, how='inner', op='intersects')

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    area.plot(ax=ax, color='lightblue', edgecolor='black')  # Plotting study area polygons
    intersections.plot(ax=ax, facecolor='none', edgecolor='red')  # Plotting intersecting hexagon grid cells
    crash_points.plot(ax=ax, color='blue', markersize=5)  # Plotting crash points
    plt.title('Intersecting Grids and Crash Points within Study Area')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def create_cell_heatmap(area_file, crash_file, num_rows=30, num_cols=30):
    """
    This function creates a heatmap of crash points density within a study area by dividing it into a grid
    of specified dimensions and counting the number of crash points within each grid cell.

    Parameters:
        area_file (str): File path to the shapefile containing the study area polygons.
        crash_file (str): File path to the shapefile containing crash points.
        num_rows (int): Number of rows for the grid.
        num_cols (int): Number of columns for the grid.
    """
    

    # Read area polygons and crash points data
    area = gpd.read_file(area_file)
    crash_points = gpd.read_file(crash_file)

    # Ensure all geometry objects have the same CRS
    crash_points = crash_points.to_crs(area.crs)

    # Ensure all geometry objects have the same CRS
    crash_points = crash_points.to_crs(area.crs)

    # Define bounding box of the study area
    xmin, ymin, xmax, ymax = area.total_bounds

    # Calculate grid cell dimensions
    grid_width = (xmax - xmin) / num_cols
    grid_height = (ymax - ymin) / num_rows

    # Create grid polygons
    grid_polygons = []
    for i in range(num_rows):
        for j in range(num_cols):
            x1 = xmin + j * grid_width
            y1 = ymin + i * grid_height
            x2 = x1 + grid_width
            y2 = y1 + grid_height
            grid_polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    # Convert grid polygons to GeoDataFrame and set CRS
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs=area.crs)

    # Add index to grid cells
    grid_gdf['grid_id'] = range(1, len(grid_gdf) + 1)

    # Spatial join crash points with grid cells and count points within each grid
    points_within_grid = gpd.sjoin(crash_points, grid_gdf, how='left', op='within')
    grid_point_counts = points_within_grid.groupby('grid_id').size().reset_index(name='point_count')

    # Merge count data with grid data
    grid_gdf = grid_gdf.merge(grid_point_counts, on='grid_id', how='left').fillna(0)

    # Find grid cells intersecting with area polygons
    intersections = gpd.sjoin(grid_gdf, area, op='intersects')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    intersections.plot(column='point_count', cmap='Blues', edgecolor='black', linewidth=0.5, ax=ax, legend=True)
    area.plot(ax=ax, color='none', edgecolor='black')
    plt.title('Heatmap of Grids based on Crash Point Count')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()



def create_hexagonal_heatmap(area_file, crash_file, resolution=8):
    """
    This function creates a heatmap of crash points density within a study area by dividing it into a hexagonal grid
    of specified resolution and counting the number of crash points within each hexagon.

    Parameters:
        area_file (str): File path to the shapefile containing the study area polygons.
        crash_file (str): File path to the shapefile containing crash points.
        resolution (int): Resolution of the H3 hexagons.
    """
    

    # Read area polygons and crash points data
    area = gpd.read_file(area_file)
    crash_points = gpd.read_file(crash_file)

    # Ensure all geometry objects have the same CRS
    crash_points = crash_points.to_crs(area.crs)

    # Get bounding box of the study area
    minx, miny, maxx, maxy = area.total_bounds

    # Convert bounding box to H3 hexagons
    hexagons = h3.polyfill_geojson(
        {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]},
        resolution
    )

    # Convert H3 hexagons to polygons
    hex_polygons = [Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True)) for hex_id in hexagons]

    # Convert polygons to GeoDataFrame
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons)

    # Spatial join crash points with hexagons and count points within each hexagon
    points_within_hex = gpd.sjoin(crash_points, hex_gdf, how='left', op='within')
    hex_point_counts = points_within_hex.groupby('index_right').size().reset_index(name='point_count')

    # Merge count data with hexagon data
    intersections_h3 = hex_gdf.merge(hex_point_counts, left_index=True, right_on='index_right', how='left').fillna(0)

    # Spatial join hexagons with area polygons
    intersections = intersections_h3.drop('index_right', axis=1)
    intersections = gpd.sjoin(intersections, area, op='intersects')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    intersections.plot(column='point_count', cmap='Blues', edgecolor='black', linewidth=0.5, ax=ax, legend=True)
    area.plot(ax=ax, color='none', edgecolor='black')
    plt.title('Heatmap of Hexagonal Grids based on Crash Point Count')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()





# Example usage
# create_cell_zones('./data/area.shp', './data/crash.shp')

# Example usage
# create_hex_grid_zones('./data/area.shp', './data/crash.shp')

# Example usage
# create_cell_heatmap('./data/area.shp', './data/crash.shp')

# Example usage
# create_hexagonal_heatmap('./data/area.shp', './data/crash.shp')