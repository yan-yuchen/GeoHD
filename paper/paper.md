---
title: 'GeoHD: A Python toolkit for geospatial hotspot detection, visualization, and analysis'
tags:
  - Python
  - GIS
  - hotspot detection
  - visualization
  - geographic analysis
authors:
  - name: Yuchen Yan^[Corresponding author]
    orcid: 0000-0002-0245-2241
    equal-contrib: true
    affiliation: "1" 
- name: Yuxin Wang
   orcid: 0009-0004-7063-3935
   affiliation: "1"
affiliations:
 - name: School of Transportation Science and Engineering, Harbin Institute of Technology, China
   index: 1
date: 10 April 2024
bibliography: paper.bib

---

# Summary

In the field of urban data analysis, the spatial distribution of research objects often exhibits non-uniformity, characterized by distinct spatial heterogeneity. Areas with higher densities of point data often signify hotspots of events within those regions. Consequently, detecting hotspots within urban areas has become a focal point in urban research, holding significant value for urban planners, researchers, and management authorities. Taking crime hotspot detection as an example, analyzing historical crime data can unveil the underlying causes of criminal activities, thereby aiding relevant management authorities in formulating more effective crime prevention strategies. In past research, various classical clustering algorithms or hotspot analysis methods such as Getis-Ord spatial statistics, k-means clustering, and kernel density analysis have been applied to urban hotspot detection. Furthermore, tailored hotspot detection approaches have been proposed by scholars for specific research contexts such as transportation, traffic accidents, and crime. Despite the emergence of excellent context-specific hotspot visualization tools like transbigdata based on transportation, challenges persist in current hotspot detection research. Existing studies often remain confined to singular research contexts, lacking comprehensive hotspot detection analysis frameworks. Additionally, current geographic processing tools primarily focus on heatmap visualization, impeding precise hotspot localization. Given the complexity of geographic spatial data, researchers from different backgrounds may encounter operational difficulties, exacerbated by the considerable learning curve associated with prevalent geographic information system software like ArcGIS. Hence, there is an urgent need to develop a universally applicable and user-friendly open-source hotspot detection tool to meet the diverse analytical needs of researchers in urban hotspot analysis.

# Statement of need


# Use Case




# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We would like to acknowledge the comments and insights from the editors and reviewers that helped lift the quality of the project and article.

# References