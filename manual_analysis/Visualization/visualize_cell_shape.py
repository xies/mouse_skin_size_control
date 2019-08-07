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
dirname = '/Users/xies/Box/Mouse/Skin/W-R2/ROIs/'

dx = 0.25

cell2plot = int(462)

# Load px, py, framne, zpos as exported by export_ROIs.py
px = []
with open(path.join(dirname,'polygon_x.csv')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for x in reader:
        x = np.array([int(a) for a in x])
        px.append(x)
        
py = []
with open(path.join(dirname,'polygon_y.csv')) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for y in reader:
        y = np.array([int(a) for a in y])
        py.append(y)

frame = np.genfromtxt(path.join(dirname,'frame.csv'), delimiter=',',dtype=np.int) - 1
zpos = np.genfromtxt(path.join(dirname,'zpos.csv'), delimiter=',',dtype=np.int) - 1
cellIDs = np.genfromtxt(path.join(dirname,'cellIDs.csv'), delimiter=',',dtype=np.int)

idx = np.where(cellIDs == cell2plot)[0]

# Zip all ROIs belonging to this cell ONLY
px = px[idx[0]:idx[-1] + 1]
py = py[idx[0]:idx[-1] + 1]
frame = frame[idx]
zpos = zpos[idx]

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


ROIs = zip(px,py,frame,zpos)
Nframes = len(np.unique(frame))
for i,t in enumerate(np.unique(frame)):
    ROIs_in_frame = [roi for roi in ROIs if roi[2] == t]
    fig = plt.figure()
    ax = a3.Axes3D(fig)
    
    for roi in ROIs_in_frame:
        # Concat all vertices
        x = roi[0]; x = (x - x.mean())*dx
        y = roi[1]; y = (y - y.mean())*dx
        z = roi[3]; z = -z + zmin
        z = np.ones(x.shape) * z
        verts = [list(zip(x,y,z))]
        # Plot polygons
        poly = a3.art3d.Poly3DCollection(verts)
        poly.set_color(None)
        poly.set_edgecolor('b')
        ax.add_collection3d(poly)
        
        # Use scatter plot to plot vertices
        ax.scatter(x,y,z,c='r')

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_zlim(-14,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Frame '+ str(t))
    plt.show()
    



