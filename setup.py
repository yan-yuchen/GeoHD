from setuptools import setup, find_packages

setup(
    name='GeoHD',
    version='0.1.5',
    description='A Python toolkit for geospatial hotspot detection, Avisualization, and analysis using urban data',
    author='Yuchen Yan',
    author_email='ycyan001@gmail.com',
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