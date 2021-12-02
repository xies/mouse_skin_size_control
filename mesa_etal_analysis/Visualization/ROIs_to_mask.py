#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:19:34 2021

@author: xies
"""

# from mpl_toolkits.mplot3d import Axes3D, art3d
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.collections import PolyCollection
# from matplotlib.patches import Polygon
import csv
from skimage import draw

X = 460; Y = 460
Z = 72

#%% Select region folder
dirname = '/Users/xies/Box/Mouse/Skin/Mesa et al/W-R1/ROIs/'

dx = 0.25

cell2plot = int(341)

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

frames = np.genfromtxt(path.join(dirname,'frame.csv'), delimiter=',',dtype=int) - 1
zpos = np.genfromtxt(path.join(dirname,'zpos.csv'), delimiter=',',dtype=int) - 1
cellIDs = np.genfromtxt(path.join(dirname,'cellIDs.csv'), delimiter=',',dtype=int)

#%%

shape = [15,Z,Y,X]
mask = np.zeros(shape)

for i,t in enumerate(frames):
    x = px[i]
    y = px[i]
    z = zpos[i]
    
    mask[t,z,:,:] = mask[t,z,:,:] + draw.polygon2mask([X,Y],np.array([x,y]).T)* cellIDs[i]
    
    
io.imsave(path.join(dirname,'polygon_mask.tif'),mask.astype(np.int16))

# ROIs = zip(px,py,frame,zpos)
# Nframes = len(np.unique(frame))
# for i,t in enumerate(np.unique(frame)):
#     ROIs_in_frame = [roi for roi in ROIs if roi[2] == t]
#     fig = plt.figure()
#     ax = Axes3D(fig)
    
#     for roi in ROIs_in_frame:
#         # Concat all vertices
#         x = roi[0]; x = (x - x.mean())*dx
#         y = roi[1]; y = (y - y.mean())*dx
#         z = roi[3]; z = -z + zmin
#         z = np.ones(x.shape) * z
#         verts = [list(zip(x,y,z))]
#         # Plot polygons
#         poly = art3d.Poly3DCollection(verts)
#         poly.set_color(None)
#         poly.set_edgecolor('b')
#         ax.add_collection3d(poly)
        
#         # Use scatter plot to plot vertices
#         ax.scatter(x,y,z,c='r')

#     ax.set_xlim(-7.5, 7.5)
#     ax.set_ylim(-7.5, 7.5)
#     ax.set_zlim(-14,1)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.title('Frame '+ str(t))
#     plt.show()
    

