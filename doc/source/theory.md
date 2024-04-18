# Theory

## Background

In the field of urban data analysis, the spatial distribution of research objects often exhibits non-uniformity, characterized by distinct spatial heterogeneity . Areas with higher densities of point data often signify hotspots of events within those regions. Consequently, detecting hotspots within urban areas has become a focal point in urban research, holding significant value for urban planners, researchers, and management authorities. Taking crime hotspot detection as an example, analyzing historical crime data can unveil the underlying causes of criminal activities, thereby aiding relevant management authorities in formulating more effective crime prevention strategies. In past research, various classical clustering algorithms or hotspot analysis methods such as Getis-Ord spatial statistics , k-means clustering , and kernel density analysis  have been applied to urban hotspot detection. Density-based hotspot detection is an important theory and method in urban research, which realizes the extraction of local research hotspots by combining density analysis and raster algebra. It has been widely used in different fields such as transportation, culture, climate, ecology and so on:

* Transportation ([Yan, Yuchen, et al. (2024)](https://doi.org/10.1111/tgis.13137))
* Housing Submarkets ([Liu, Xinrui, et al.  (2021)](https://doi.org/10.1155/2022/2948352))
* Culture ([Zhang, Haiping, et al.  (2021)](https://doi.org/10.1111/tgis.12682))
* Ecology ([Qian, Chunhua, et al.  (2021)](https://doi.org/10.1177/15501477211039137))

Existing studies often remain confined to singular research contexts, lacking comprehensive hotspot detection analysis frameworks. Additionally, current geographic processing tools primarily focus on heatmap visualization, impeding precise hotspot localization. Given the complexity of geographic spatial data, researchers from different backgrounds may encounter operational difficulties, exacerbated by the considerable learning curve associated with prevalent geographic information system software like ArcGIS. Hence, there is an urgent need to develop a universally applicable and user-friendly open-source hotspot detection tool to meet the diverse analytical needs of researchers in urban hotspot analysis.

## Geospatial hotspot detection

GeoHD is a Python toolbox designed for the detection, visualization, and analysis of geographical spatial hotspots. Its primary objective is to provide a user-friendly tool applicable across various urban research backgrounds for hotspot detection and analysis.


