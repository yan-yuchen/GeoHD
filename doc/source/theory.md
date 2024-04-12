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

### Adaptive Kernel Density Estimation

The Adaptive Kernel Density Estimation (AKDE) is a technique tailored to adjust the bandwidth of Kernel Density Estimation (KDE) in accordance with the spatial arrangement of disparate datasets. This research refines the AKDE for pinpointing hotspots within urban confines through a trio of sequential stages: preliminary assessment, local bandwidth specification, and computation of kernel density.

#### Step 1: Initial Assessment

The initial assessment phase entails creating an initial approximation of the distribution of the event under study, denoted as $S$, by employing kernel density estimation with an assigned bandwidth. A Gaussian function is utilized as the core function for this preliminary approximation. Given a dataset comprising $N$ points within the research area, with each event $s_i$ located at coordinates $\langle x_i, y_i \rangle$, the initial estimate of the density $f_{\tilde{HS}}(s)$ is derived as follows:

$$f_{\tilde{HS}}(s) = \frac{1}{N} \sum_{i=1}^{N} K_{H}(s - s_i) \tag{1}$$

The core function $K_{H}(s - s_i)$ is a Gaussian with bandwidth $H$:

$$K_{H}(s - s_i) = \frac{1}{2\pi H_1 H_2} \exp\left(-\frac{(x - x_i)^2}{2H_1^2} - \frac{(y - y_i)^2}{2H_2^2}\right) \tag{2}$$

The bandwidth $H$ is composed of the longitudinal bandwidth $H_1$ and the latitudinal bandwidth $H_2$:

$$H_1 = 1.06N^{-1/5} \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \frac{1}{N} \sum_{j=1}^{N} x_j)^2} \tag{3}$$

$$H_2 = 1.06N^{-1/5} \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \frac{1}{N} \sum_{j=1}^{N} y_j)^2} \tag{4}$$

#### Step 2: Specification of Local Bandwidth

Utilizing the outcomes from the initial assessment, the local bandwidth at $s_i$, denoted as $h_i$, is ascertained:

$$h_i = \left(\left(g^{-1} f_{\tilde{HS}}(s_i)\right)^{-\alpha}\right) \tag{5}$$

The parameter $\alpha$ is a sensitivity factor that ranges from 0 to 1, inclusive. A higher value of $\alpha$ results in more pronounced variations in local bandwidths. The geometric mean $g$ is defined as:

$$g = \sqrt{n \prod_{i=1}^{n} f_{\tilde{HS}}(s_i)} \tag{6}$$

#### Step 3: Computation of Kernel Density

The AKDE is ultimately computed using the local bandwidths:

$$f_{HS}(s) = \frac{1}{N} \sum_{i=1}^{N} K_{H_{h_i}}(s - s_i) \tag{7}$$

$$K_{H_{h_i}}(s - s_i) = \frac{1}{2\pi H_1 H_2 h_i^2} \exp\left(-\frac{(x - x_i)^2}{2H_1^2 h_i^2} - \frac{(y - y_i)^2}{2H_2^2 h_i^2}\right) \tag{8}$$

In areas characterized by a dense concentration of events, the larger initial estimate from Eq. (1) used in Eq. (5) will lead to a reduced local bandwidth. In contrast, areas with a sparser distribution of events will yield an expanded local bandwidth.

Incorporating AKDE in this investigation facilitates the circumvention of estimation biases associated with a fixed bandwidth and enhances the precision of spatial distribution variations in event occurrences.

### Hotspot Recognition

The process of identifying local hotspots of events within a study area involves a series of computational steps. Initially, we examine the dataset `S` consisting of events that have occurred within the area of interest, represented as `S = {s_1, s_2, ..., s_N}`. Following an adaptive kernel density analysis, we generate a raster dataset that depicts the density field. We then proceed with further manipulations of this raster data. Utilizing a window analysis technique, we compute a pixel maximum value surface from the raster data. Subsequently, we engage in a map algebraic operation, subtracting the pixel maximum value surface from the density field surface to create a subtracted surface, which is a non-negative value surface. Locations with zero values in the subtracted surface indicate the positions of local hotspots in the current area. After the window analysis, we employ a reclassification technique to segregate the extreme value regions from the non-extreme value areas into two distinct classes. The extreme value regions, identified through this method, correspond to the hotspots of events in the current area, denoted as `HS = {hs_1, hs_2, ..., hs_K}`. The specific steps of this methodology are as follows:

#### Step 1: Data Input
Introduce the dataset `S`, which contains events from the study area, into the computational system.

#### Step 2: Adaptive Kernel Density Analysis
Subject the dataset `S` to an adaptive kernel density analysis to produce a raster dataset representing the density field. This dataset mirrors the spatial arrangement of events within the study area.

#### Step 3: Window Analysis Method
Implement the window analysis method to calculate the pixel maximum value surface from the raster data. This step yields the highest density value within each defined window.

#### Step 4: Map Algebra Subtraction Operation
Execute a map algebraic subtraction between the density field surface and the pixel maximum value surface to generate a subtracted surface. Locations within each window that exhibit zero values are indicative of local hotspots in the current area.

#### Step 5: Reclassification Algorithm
Apply a reclassification algorithm to divide the subtracted surface into two categories: regions with extreme values and regions without extreme values.

#### Step 6: Hotspot Extraction
Isolate the regions with extreme values from the reclassified dataset to extract the set of hotspots `HS`, which correspond to areas of event occurrence within the current area.

