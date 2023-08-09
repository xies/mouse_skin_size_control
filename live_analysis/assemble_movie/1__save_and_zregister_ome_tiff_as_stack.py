#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:16:59 2022

@author: xies
"""

import numpy as np
from skimage import io, util
from os import path
from glob import glob
from pystackreg import StackReg
from re import findall
from tqdm import tqdm

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/'

OVERWRITE = False

#%% Reading the first ome-tiff file using imread reads entire stack

# Extract the first ome.tiff file from every subfolder, load, then separate the two channels
def sort_by_slice(filename):
    z = findall('_(\d+).ome.tif',filename)[0]
    return int(z)

subfolders = glob(path.join(dirname,'R*/*/ZSeries*/'))

header_ome_h2b = []
header_ome_fucci = []
for d in subfolders:
    ome_tifs = glob(path.join(d,'*.ome.tif'))
    ome_tifs = sorted(ome_tifs) # Sort by channel #
    ome_tifs = sorted(ome_tifs, key = sort_by_slice) # Sort by slice #
    if len(ome_tifs) < 40:
        print(f'Skipping {d}')
    else:
        if len(findall('1020nm',path.split(path.split(d)[0])[1])) == 0:
            header_ome_h2b.append(ome_tifs[0])
        else:
            header_ome_fucci.append(ome_tifs[0])

#%% Register the B/G channels (using B as reference)

channel_names = ['G','B']
for header_ome in tqdm(header_ome_h2b):
    
    d = path.split(path.dirname(header_ome))[0]
    # Make sure we haven't already processed this stack
    if path.exists(path.join(d,'G_reg.tif')) and not OVERWRITE:
        print(f'Skipping {d}')
        continue
    
    # Load ome-tif
    print(f'Loading {d}')
    stack = io.imread(header_ome,is_ome=True)
    if stack.ndim > 3:
        G = stack[0,...]
        B = stack[1,...]
    else:
        B = stack
    
    # Use StackReg
    print(f'Registering {d}')
    sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
    T = sr.register_stack(B,reference='previous',n_frames=20,axis=0) #Obtain the transformation matrices
    B_reg = sr.transform_stack(B,tmats=T) # Apply to both channels
    G_reg = sr.transform_stack(G,tmats=T)
    
    output_path = path.join( d,'B_reg.tif')
    io.imsave(output_path,util.img_as_uint(B_reg/B_reg.max()),check_contrast=False)
    output_path = path.join( d,'G_reg.tif')
    io.imsave(output_path,util.img_as_uint(G_reg/G_reg.max()),check_contrast=False)
    
    print(f'Saved with {output_path}')

#%% Register the FUCCI (R) channels (Using R)

channel_names = ['R','R_shg']
for header_ome in tqdm(header_ome_fucci):

    d = path.split(path.dirname(header_ome))[0]
    # Make sure we haven't already processed this stack
    if path.exists(path.join(d,'R_reg.tif')) and not OVERWRITE:
        print(f'Skipping {d}')
        continue
    
    # Load ome-tif
    print(f'Loading {d}')
    stack = io.imread(header_ome)
    R = stack[0,...]
    R_shg = stack[1,...]
    
    # Use StackReg
    print(f'Registering {d}')
    sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
    T = sr.register_stack(R_shg,reference='previous',axis=0) #Obtain the transformation matrices
    R_reg = sr.transform_stack(R,tmats=T) # Apply to both channels
    R_shg_reg = sr.transform_stack(R_shg,tmats=T) # Apply to both channels
    
    output_path = path.join( d,'R_reg.tif')
    io.imsave(output_path,util.img_as_uint(R_reg/R_reg.max()),check_contrast=False)
    output_path = path.join( d,'R_shg_reg.tif')
    io.imsave(output_path,util.img_as_uint(R_shg_reg/R_shg_reg.max()),check_contrast=False)
    
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

