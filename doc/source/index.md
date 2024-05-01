# GeoHD

![python](https://img.shields.io/badge/python-3.11-black)
![GitHub release](https://img.shields.io/badge/release-v0.2.7-blue)
![pypi](https://img.shields.io/badge/pypi-v0.2.7-orange)
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
## Issues

If you encounter any **bugs** or **problems** with GeoHD, please create a post using our package [issue tracker](https://github.com/yan-yuchen/GeoHD/issues). Please provide a clear and concise description of the problem, with images or code-snippets where appropriate. We will do our best to address these problems as fast and efficiently as possible.

## Contribute

The authors of the GeoHD package welcome external contributions to the source code. This process will be easiest if users adhere to the contribution policy:

* Open an issue on the package [issue tracker](https://github.com/yan-yuchen/GeoHD/issues) clearly describing your intentions on code modifications or additions
* Ensure your modifications or additions adhere to the existing standard of the GeoHD package, specifically detailed documentation for new methods (see existing methods for example documentation)
* Test your modifications to ensure that the core functionality of the package has not been altered by running the unit tests 
* Once the issue has been discussed with a package author, you may open a pull request containing your modifications

## Citation

Yan, Y., 2024. GeoHD: A Python Toolkit for Geospatial Hotspot Detection, Visualization, and Analysis. SSRN Electron. J. https://doi.org/10.2139/ssrn.4799686

Yan, Y., Quan, W., Wang, H., 2024. A data‐driven adaptive geospatial hotspot detection approach in smart cities. Trans. GIS tgis.13137. https://doi.org/10.1111/tgis.13137

## Authors

* Yuchen Yan


```{toctree}
---
maxdepth: 2
caption: Documentation
hidden: true
---
getting_started
theory
api
changelog
GitHub <https://github.com/yan-yuchen/GeoHD>
```