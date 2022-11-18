#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:57:06 2022

@author: xies
"""

import numpy as np
import os.path as path
import pickle as pkl
import csv
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from glob import glob

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/tracked_cells/'
# dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/F2 WT/R2/manual_track'
# dirname = '/Users/xies/Desktop/KO R2/manual_track'

#%%

filename = glob(path.join(dirname,'*/*[ab].xpts.txt'))

# Draw PolyRoi using PIL.ImageDraw
X = 1024
T = 20
Z = 68
# Create blank images as list of lists (can't use ndimage for PIL)
# Images are passed as reference
im_tzstack = []
for t in range(T):
    im_zstack = []
    for z in range(Z):
        im_zstack.append(Image.new(mode='1',size=(X,X)))
        
    im_tzstack.append(im_zstack)

for f in filename:
    
    # Load px, py, framne, zpos as exported by export_ROIs.py
    basename = path.splitext(path.splitext(f)[0])[0]
    
    cellID = int(path.basename(basename).split('.')[0])
    
    px = []
    with open(f) as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for x in reader:
            x = np.array([int(a) for a in x])
            px.append(x)
    py = []
    with open(basename + '.ypts.txt') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for y in reader:
            y = np.array([int(a) for a in y])
            py.append(y)    
        
    zpos = np.genfromtxt(basename + '.zpts.txt', delimiter=',',dtype=np.int) -1
    frame = [int( path.basename(basename).split('.')[1])-1] * len(px)
    
    ROIs = list(zip(px,py,frame,zpos))

    Npoly = len(ROIs)

    for roi in ROIs:
        x = roi[0]; y = roi[1]
    #    print frame[i], zpos[i]
        im = im_tzstack[roi[2]][roi[3]] # Recorded Frame is +1 from list position
        draw = ImageDraw.Draw(im)
        # xy = []
        draw.polygon( list(zip(x,y)) , outline='white')
        del draw
        
    #     # Find centroid of cell and print CellID inside
    #     shape = Polygon( np.vstack( ( roi[0],roi[1] )).T)
    #     A = shape.area
    #     cx = shape.centroid.coords.xy[0][0]
    #     cy = shape.centroid.coords.xy[1][0]
    #     draw = ImageDraw.Draw(im)
    #     draw.text([cx-5,cy-5],str(roi[4]),fill=225)
    #     del draw

#%%
# And then draw CellID on the first frame
verdana = ImageFont.truetype('/Library/Fonts/Verdana.ttf',size=8)

#for c in collated:
#    t = c.iloc[0].Frame
#    cellID = c.iloc[0].CellID
#    I = [ roi[4] == cellID for roi in ROIs] # Check if current cell has track... it should...
#    I = np.where(I)[0]
#    for i in I:
#        if ROIs[i][2] == t: # Check first frame
#            roi = ROIs[i]
#            # Find centroid of cell and print CellID inside
#            shape = Polygon( np.vstack( ( roi[0],roi[1] )).T)
#            A = shape.area
#            cx = shape.centroid.coords.xy[0][0]
#            cy = shape.centroid.coords.xy[1][0]
#            im = im_tzstack[t-1][roi[3]]
#            draw = ImageDraw.Draw(im)
#            draw.text([cx-5,cy-5],str(cellID),fill=225)
#            del draw
            
        
for t in range(T):
    for z in range(Z):
        filebase = 'mask_t%03d_z%03d.tif' % (t,z);
        filename = path.join(dirname,filebase)
        im = im_tzstack[t][z]
        im.save( filename, format='tiff')
        
        