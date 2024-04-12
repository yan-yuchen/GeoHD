# Getting started

## Install with pip

The package is available in PyPi and requires [Python 3.11](https://www.python.org/downloads/) or higher. It can be installed using:

```bash
$ pip install GeoHD
```

## Basic Usage

Visualization of hotspots on real maps:

```python
visualize_shapefile('data.shp', output_image_path='custom_image.png')
```


Analytic Plane Point Patterns: Ripley G, Ripley F, Ripley J, Ripley K, Ripley L, etc. through the plotting function.

```python
plot_g_function('data.shp')
```


The study area was divided into a quadrilateral (hexagonal) grid and fast visualization was achieved based on the density of point data within the divided area.

```python
create_cell_zones(area_file, crash_file)
create_hex_grid_zones(area_file, crash_file)
create_cell_heatmap(area_file, crash_file)
create_hexagonal_heatmap(area_file, crash_file)
```


Realization of kernel density analysis with fixed bandwidth:

```python
density_raster = process_shapefile(input_file_path)
plot_density_raster(density_raster,output_data_path, *gpd.read_file(input_file_path).total_bounds)
```

Kernel density analysis for realizing adaptive bandwidth:

```python
adaptiveKDE(shp_file,output_data_path)
```


Hotspot Identification:

```python
hotspots = extract_hotspots(density_data_path)
visualize_hotspots(np.load(density_data_path), hotspots)
```

