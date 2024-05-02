#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:01:42 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from skimage import io, measure, segmentation
from scipy.ndimage import distance_transform_edt
from glob import glob
from os import path
from tqdm import tqdm

from pyvoro import compute_voronoi
from re import match

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%% Load 3D segs and generate voronoi

# for t in range(15):
t = 0 

im = io.imread(path.join(dirname,f'Image flattening/flat_3d_nuc_seg/t{t}.tif'))

# find centroids for each item
df = pd.DataFrame( measure.regionprops_table(im,properties=['label','centroid']) )
df = df.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X'})

bbox = [[0,72],[0,460],[0,460]]

voronois = compute_voronoi(df[['Z','Y','X']].values, limits=bbox, dispersion=20)

#%% Visualize polyhedra

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

fig = plt.figure()

vor = voronois[0]
vertices = vor['vertices']
faces = vor['faces']
polygons =[ np.array([vertices[j] for j in faces[i]['vertices']]) for i in range(len(faces))]

c = colors.rgb2hex((np.random.rand(3)))

def plot_polygon3d(ax,polygon,color='blue'):
    # Polygon -> vertex list arranged counterclockwise
    f = a3.art3d.Poly3DCollection([polygon])
    f.set_color(c)
    f.set_edgecolor('k')
    f.set_alpha(0.5)
    ax.add_collection3d(f)

ax = fig.add_subplot(111, projection="3d")
ax.azim=-140
ax.elev=20

ax.set_xlim([0,76])
ax.set_ylim([-1,460])
ax.set_zlim([-1,460])

vertices = vor['vertices']
faces = vor['faces']
    
for i,face in enumerate(faces):
    
    polygon = [vertices[j] for j in faces[i]['vertices']]
    
    plot_polygon3d(ax,polygon,color=c)

plt.show()

#%% Generate 'segmentation' based on Voronoi

vor = voronois[0]
vertices = vor['vertices']
faces = vor['faces']
polygons =[ np.array([vertices[j] for j in faces[i]['vertices']]) for i in range(len(faces))]

def get_surface_normal(polygon):
    
    Nverts = polygon.shape[0]
    assert(Nverts > 2) # at least a triangle
    
    root = 0
    plusone = (root + 1) % Nverts
    minusone = (root - 1) % Nverts
    
    a = polygon[0,:] - polygon[1,:]
    b = polygon[2,:] - polygon[1,:]
    normal = np.cross(a,b)
    normal = normal / np.linalg.norm(normal)
    return normal

def plot_normal_and_test_vectors(ax,point,polygon):

    # plot_polygon3d(ax,polygon)
    normal = get_surface_normal(polygon)
    ax.quiver(point[0],point[1],point[2],normal[0],normal[1],normal[2])
    vec = -(point - polygon[0,:])
    vec = vec / np.linalg.norm(vec)
    ax.quiver(point[0],point[1],point[2],vec[0],vec[1],vec[2],color='r')

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.azim=-140
ax.elev=20

ax.set_xlim([0,76])
ax.set_ylim([-1,460])
ax.set_zlim([-1,460])

def point_is_in_polyhedron(point,polygons,ax=None):
    # print(point)
    on_negative_side = np.zeros(len(polygons))

    for i,polygon in enumerate(polygons):
        
        # plot_polygon3d(ax,polygon,color=c)
        
        normal = get_surface_normal(polygon)
        
        # Get displacement vector from point to any vertex on polygon
        test_vector = point - polygon[0,:]
        
        on_negative_side[i] = np.dot(normal,test_vector) < 0
        
    return np.all(on_negative_side)


#%%

vor_segmentation = np.zeros([25,460,460])

for i,vor in tqdm(enumerate(voronois)):
    
    label = i + 1
    
    vertices = vor['vertices']
    faces = vor['faces']
    polygons =[ np.array([vertices[j] for j in faces[k]['vertices']]) for k in range(len(faces))]
    
    maxes = np.ceil(np.array(vertices).max(axis=0)).astype(int)
    mins = np.floor(np.array(vertices).min(axis=0)).astype(int)
    
    Z = np.arange( max(0,mins[0]), min(19,maxes[0]))
    Y = np.arange( max(0,mins[1]), min(459,maxes[1]))
    X = np.arange( max(0,mins[2]), min(459,maxes[2]))
    
    ZZ,YY,XX = np.meshgrid(Z,Y,X,sparse=False)
    
    coords2test = list(zip(ZZ.flatten(),YY.flatten(),XX.flatten()))
    
    for coord in tqdm(coords2test):
        if vor_segmentation[coord[0],coord[1],coord[2]] == 0:
            vor_segmentation[coord[0],coord[1],coord[2]] = point_is_in_polyhedron(coord,polygons) * label



io.imsave(path.join(dirname,'voro_seg.tif'),vor_segmentation)


