from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GeoHD',
    version='0.1.7',
    description='A Python toolkit for geospatial hotspot detection, Avisualization, and analysis using urban data',
    author='Yuchen Yan',
    author_email='ycyan001@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yan-yuchen/GeoHD",    
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'contextily',
        'geopandas',
        'pointpats',
        'h3',
    ],
)