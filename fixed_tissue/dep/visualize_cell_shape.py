#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:21:12 2019

@author: xies
"""

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from os import path
import csv

# Select region folder
dirname = '/Users/xies/Box/Mouse/Skin/Fixed/11-06-2019 Skin Ecad488 EdU8h/WT/3/'

dx = 0.1660532

# Load px, py, framne, zpos as exported by export_ROIs.py
px = []
with open(path.join(dirname,'3.xpts.txt')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for x in reader:
        x = np.array([int(a) for a in x])
        px.append(x)
        
py = []
with open(path.join(dirname,'3.ypts.txt')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for y in reader:
        y = np.array([int(a) for a in y])
        py.append(y)

zpos = np.genfromtxt(path.join(dirname,'3.zpts.txt'), delimiter=',',dtype=np.int) - 1

# Flip z positions to be right side up
# Set axes limits
xmin = min([ max(x) for x in px ])
xmax = max([ max(x) for x in px ])
x_range = xmax - xmin
ymin = min([ min(y) for y in py ])
ymax = max([ max(y) for y in py ])
y_range = ymax - ymin
zmin = min(zpos)
zmax = max(zpos)
z_range = zmax-zmin

ROIs = zip(px,py,zpos)

fig = plt.figure()
ax = Axes3D(fig)

for roi in ROIs:
    # Concat all vertices
    x = roi[0];
    x = (x - xmin)*dx
    y = roi[1];
    y = (-y + ymin)*dx
    z = roi[2];
    z = -z + zmin
    z = np.ones(x.shape) * z
    verts = [list(zip(x,y,z))]
    # Plot polygons
    poly = Poly3DCollection(verts)
    poly.set_color(None)
    poly.set_edgecolor('b')
    ax.add_collection3d(poly)
    
    # Use scatter plot to plot vertices
    ax.scatter(x,y,z,c='r')
    
ax.set_xlim(-7.5, 20)
ax.set_ylim(-20, 7.5)
ax.set_zlim(-14,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

    
    
    
