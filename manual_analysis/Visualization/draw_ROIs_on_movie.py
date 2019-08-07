#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:20:57 2019

@author: xies
"""

import numpy as np
import os.path as path
import pickle as pkl
import csv
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
collated = c2

dirname = '/Users/xies/Box/Mouse/Skin/W-R1/Color ROIs/'
X = 460; Y= 460; Z = 72; T = 15;

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

ROIs = zip(px,py,frame,zpos,cellIDs)

Npoly = len(ROIs)


# Draw PolyRoi using PIL.ImageDraw

# Create blank images as list of lists (can't use ndimage for PIL)
# Images are passed as reference
im_tzstack = []
for t in range(T):
    im_zstack = []
    for z in range(Z):
        im_zstack.append(Image.new(mode='1',size=(X,Y)))
    im_tzstack.append(im_zstack)
    

for roi in ROIs:
    x = roi[0]; y = roi[1]
#    print frame[i], zpos[i]
    im = im_tzstack[roi[2]][roi[3]] # Recorded Frame is +1 from list position
    draw = ImageDraw.Draw(im)
    xy = []
    draw.polygon( zip(x,y) , outline='white')
    del draw
    
    # Find centroid of cell and print CellID inside
    shape = Polygon( np.vstack( ( roi[0],roi[1] )).T)
    A = shape.area
    cx = shape.centroid.coords.xy[0][0]
    cy = shape.centroid.coords.xy[1][0]
    draw = ImageDraw.Draw(im)
    draw.text([cx-5,cy-5],str(roi[4]),fill=225)
    del draw

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
        filename = path.join(dirname,'images/',filebase)
        im = im_tzstack[t][z]
        im.save( filename, format='tiff')
        
        