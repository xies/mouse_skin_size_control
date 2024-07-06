#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:19:34 2021

@author: xies
"""

from os import path
import csv
from skimage import draw, io

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

#%% Make mask

shape = [15,Z,Y,X]
mask = np.zeros(shape)

for i,t in enumerate(frames):
    x = px[i]
    y = py[i]
    z = zpos[i]
    
    rr,cc = draw.polygon(x,y,shape=[X,Y])
    
    mask[t,z,rr,cc] = cellIDs[i]
    
    
io.imsave(path.join('/Users/xies/Desktop/','polygon_mask.tif'),mask.astype(np.int16))

