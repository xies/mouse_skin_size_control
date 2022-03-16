#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:16:59 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters, util
from os import path
from glob import glob
from pystackreg import StackReg
from re import findall


dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/03-03-2022/M8 WT/R3 pw 250/'

#%% Reading the first ome-tiff file using imread reads entire stack

# Extract the first ome.tiff file from every subfolder, load, then separate the two channels

subfolders = glob(path.join(dirname,'Day*/ZSeries*/'))

channel_names = ['G','B']

header_ome_h2b = []
header_ome_fucci = []
for d in subfolders:
    ome_tifs = glob(path.join(d,'*.tif'))
    if len(ome_tifs) < 30:
        print(f'Skipping {d}')
    else:
        if len(findall('1020nm',d)) == 0:
            header_ome_h2b.append(ome_tifs[0])
        else:
            header_ome_fucci.append(ome_tifs[0])


#%% Register the B/G channels (using B as reference)

for header_ome in header_ome_h2b:
    
    # Load ome-tif
    stack = io.imread(header_ome)
    G = stack[0,...]
    B = stack[1,...]
    
    # Use StackReg
    sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
    T = sr.register_stack(B,reference='previous',n_frames=20,axis=0) #Obtain the transformation matrices
    B_reg = sr.transform_stack(B,tmats=T) # Apply to both channels
    G_reg = sr.transform_stack(G,tmats=T)
    
    output_path = path.join( path.dirname(header_ome),'B_reg.tif')
    io.imsave(output_path,B_reg.astype(np.int16))
    output_path = path.join( path.dirname(header_ome),'G_reg.tif')
    io.imsave(output_path,G_reg.astype(np.int16))
    
    print(f'Saved with {output_path}')

#%% Parse xml file into a dataframe organized by channel (column) and Frame (row)
#NB: not needed for now, but working
# xmllist = glob(path.join(dirname,'Day*/Z*/*.xml'))

# def extract_filenames_from_xml(path_to_xml,channel_names):
    
#     from xml.etree import ElementTree as et
    
#     tree = et.parse(path_to_xml)
#     root = tree.getroot()
    
#     # Get the sequence (should be only one)
#     seq = root.findall('Sequence')
#     assert(len(seq) == 1)
    
#     df = pd.DataFrame()
#     # Get eachframe
#     frames = seq[0].findall('Frame')
#     assert(len(frames) > 0)
    
#     for i in range(len(channel_names)):
#         framelist = []
#         for f in frames:
#             framelist.append(f.findall('File')[i].attrib['filename'])
            
#         df[channel_names[i]] = framelist
        
#     return df

# df = extract_filenames_from_xml(xmllist[0],channel_names)
# for chan in channel_names:
#     df[chan] = path.dirname(xmllist[0]) + '/' + df[chan]

