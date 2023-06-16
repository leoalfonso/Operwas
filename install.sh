#!/usr/bin/env bash

# g++:
sudo apt-get install -y g++

# python-dev:
sudo apt-get install -y python3 python3-dev python3-pip python3-tk

# GDAL:
sudo apt-get install -y libgdal-dev
sudo apt-get install -y gdal-bin
sudo apt-get install -y python3-gdal

# rtree:
sudo apt-get install -y python3-rtree

# setuptools:
pip3 install setuptools wheel

# numpy (sufficient version)
pip3 install "numpy>=1.17"

# scipy:
pip3 install scipy

# matplotlib:
pip3 install matplotlib

# pandas:
pip3 install pandas

# xlrd:
pip3 install xlrd

# pygeoprocessing:
pip3 install pygeoprocessing

# platypus:
pip3 install platypus-opt

# Pillow (PIL):
pip3 install Pillow

# pyshp (shapefile):
pip3 install pyshp

# seaborn (optional):
pip3 install seaborn
