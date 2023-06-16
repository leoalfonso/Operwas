FROM ubuntu:18.04
LABEL author="Axel Dahlberg <axel.dahlberg12@gmail.com>"

ENV HOME /root

# Update docker image
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y

# Install Python 3
RUN apt-get install -y python3 python3-dev python3-pip python3-tk

# Set a UTF-8 locale - this is needed for some python packages to play nice
RUN apt-get -y install language-pack-en
ENV LANG="en_US.UTF-8"

# Install ssh, git, vim
RUN apt-get -y install ssh git vim

# g++:
RUN apt-get install g++

# GDAL:
RUN apt-get install -y libgdal-dev
RUN apt-get install -y gdal-bin
RUN apt-get install -y python3-gdal

# rtree:
RUN apt-get install -y python3-rtree

# Upgrade pip
RUN pip3 install --upgrade pip

# setuptools:
RUN pip3 install setuptools wheel

# numpy (sufficient version)
RUN pip3 install "numpy>=1.17"

# scipy:
RUN pip3 install scipy

# matplotlib:
RUN pip3 install matplotlib

# pandas:
RUN pip3 install pandas

# xlrd:
RUN pip3 install xlrd

# pygeoprocessing:
RUN pip3 install pygeoprocessing

# platypus:
RUN pip3 install platypus-opt

# Pillow (PIL):
RUN pip3 install Pillow

# pyshp (shapefile):
RUN pip3 install pyshp

# seaborn (optional):
RUN pip3 install seaborn
