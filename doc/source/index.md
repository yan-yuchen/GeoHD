# GeoHD

![python](https://img.shields.io/badge/python-3.11-black)
![GitHub release](https://img.shields.io/badge/release-v0.2.4-blue)
![pypi](https://img.shields.io/badge/pypi-v0.2.4-orange)
![license](https://img.shields.io/badge/license-GNU%20AGPLv3-green)


## What is GeoHD? 
GeoHD is a Python toolkit for geospatial hotspot detection, visualization, and analysis using urban data. Density-based hotspot detection is an important theory and method in urban research, which realizes the extraction of local research hotspots by combining density analysis and raster algebra. It has been widely used in different fields such as transportation, culture, climate, ecology and so on:

* Transportation ([Yan, Yuchen, et al. (2024)](https://doi.org/10.1111/tgis.13137))
* Housing Submarkets ([Liu, Xinrui, et al.  (2021)](https://doi.org/10.1155/2022/2948352))
* Culture ([Zhang, Haiping, et al.  (2021)](https://doi.org/10.1111/tgis.12682))
* Ecology ([Qian, Chunhua, et al.  (2021)](https://doi.org/10.1177/15501477211039137))

The main functions of GeoHD are fast visualization and hotspot detection based on geospatial point data, and it realizes fixed-bandwidth hotspot detection with adjustable parameters and adaptive-bandwidth hotspot detection. In addition, GeoHD provides spatial point pattern distribution analysis such as Ripley G-function calculation and fast comparison image drawing.


## Install with pip

The package is available in PyPi and requires [Python 3.11](https://www.python.org/downloads/) or higher. It can be installed using:

```bash
$ pip install GeoHD
```


```{toctree}
---
maxdepth: 2
caption: Documentation
hidden: true
---
installation
getting_started
api
changelog
GitHub <https://github.com/yan-yuchen/GeoHD>
```