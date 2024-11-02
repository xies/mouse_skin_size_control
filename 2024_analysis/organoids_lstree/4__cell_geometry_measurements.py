#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:02:33 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import matplotlib.pyplot as plt
from tqdm import tqdm
from mathUtils import parse_3D_inertial_tensor, get_interpolated_curve
import seaborn as sb
import pyvista as pv
import meshFMI
import pickle as pkl
from imageUtils import most_likely_label

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2

filt_seg = io.imread(path.join(dirname,'manual_cellcycle_annotations/filtered_segs.tif'))
T = filt_seg.shape[0]

#%%

pl = pv.Plotter()
df = []
for t in tqdm(range(T)):
    
    # Everything image-wise is handled as 1-indexed
    H2B = io.imread(path.join(dirname,f'Channel0-Deconv/Channel0-T{t+1:04d}.tif'))
    Cdt1 = io.imread(path.join(dirname,f'Channel1-Denoised/Channel1-T{t+1:04d}.tif'))
    Gem = io.imread(path.join(dirname,f'Channel2-Denoised/Channel2-T{t+1:04d}.tif'))
    all_labels = io.imread(path.join(dirname,f'manual_segmentation/man_Channel0-T{t+1:04d}.tif'))
    
    props = measure.regionprops(all_labels,intensity_image=H2B, spacing=[dz,dx,dx])
    _df = pd.DataFrame(index=range(len(props)),columns=['cellID', 'Nuclear volume'
                                                        ,'Axial moment','Axial angle'
                                                        ,'Planar moment 1'
                                                        ,'Planar moment 2'
                                                        ,'Planar angle'
                                                        ,'Z','Y','X'
                                                        ,'Surface area'
                                                        ,'Mean H2B intensity'])
    
    meshes_in_frame = {}
    cell_vectors = {}
    for i,p in enumerate(props):
        _df['Frame'] = t
        _df.loc[i,'cellID'] = p['label']
        _df.loc[i,'Mean H2B intensity']= p['mean_intensity']
        _df.loc[i,'Nuclear volume px'] = p['area'] / dz / dx /dx
        _df.loc[i,'Nuclear volume'] = p['area']
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        _df.loc[i,'Axial moment'] = Iaxial
        _df.loc[i,'Axial angle'] = phi
        _df.loc[i,'Planar moment 1'] = Ia
        _df.loc[i,'Planar moment 2'] = Ib
        _df.loc[i,'Planar angle'] = theta
        
        # Also just save the principal axis
        L,D = np.linalg.eig(I)
        _df.loc[i,['Principal axis-0','Principal axis-1','Principal axis-2']] = D[:,0]
        cell_vectors[p['label']] = D[:,0]
        
        z,y,x = p['centroid']
        _df.loc[i,'Z'] = z
        _df.loc[i,'Y'] = y
        _df.loc[i,'X'] = x
        _df.loc[i,'Z - px'] = z / dz
        _df.loc[i,'Y - px'] = y / dx
        _df.loc[i,'X - px'] = x /dx
        
        if p['area'] > 55:

            # Extract cropped mask and generated mesht using VTK (meshFMI), and convert into pyVista API
            bbox = p['bbox']
            minZ,minY,minX = bbox[:3]
            maxZ,maxY,maxX = bbox[3:]
            cropped_mask = (all_labels == p['label'])[minZ - 5 : maxZ + 5,
                                                           minY - 5 : maxY + 5,
                                                           minX- 5 : maxX + 5]

            cropped_vtk = meshFMI.numpy_img_to_vtk(cropped_mask.astype(int),[dz,dx,dx],
                                                    origin=[minZ*dz,minY*dx,minX*dx], deep_copy=False)
            mesh = pv.PolyData( meshFMI.extract_smooth_mesh(cropped_vtk,[1,1]) )
            meshes_in_frame[p['label']] = mesh

            _df.loc[i,'Surface area'] = mesh.area
            _df.loc[i,'Mesh volume'] = mesh.volume
    
    # Export the meshes
    with open(path.join(dirname,f'manual_seg_mesh/individual_mesh_by_cellID_T{t+1:04d}.pkl'),'wb') as f:
         pkl.dump(meshes_in_frame,f)
    # Export the principal vectors
    _df.loc[:,['Z','Y','X',
               'Principal axis-0','Principal axis-1','Principal axis-2']].to_csv(
                   path.join(dirname,f'manual_seg_mesh/principal_vector_cellID_T{t+1:04d}.csv'))
    
    # Measure other channels + merge
    _R = pd.DataFrame(
        measure.regionprops_table(all_labels,intensity_image=Cdt1,properties=['label','mean_intensity']))
    _R = _R.rename(columns={'label':'cellID','mean_intensity':'Mean Cdt1 intensity'})
    _df = _df.merge(_R,on='cellID')

    _G = pd.DataFrame(
        measure.regionprops_table(all_labels,intensity_image=Gem,properties=['label','mean_intensity']))
    _G = _G.rename(columns={'label':'cellID','mean_intensity':'Mean Gem intensity'})
    _df = _df.merge(_G,on='cellID')

    # Measure the cell tracking to find the track label
    _tracked = pd.DataFrame(measure.regionprops_table(all_labels,intensity_image=filt_seg[t,...],
                                                      extra_properties=[most_likely_label]))
    _tracked = _tracked[['label','most_likely_label']]
    _tracked = _tracked.rename(columns={'label':'cellID','most_likely_label':'trackID'})
    _tracked.loc[_tracked['trackID'] == 0,'trackID'] = np.nan
    
    _df = pd.merge(_df,_tracked,on='cellID',how='left')

    df.append(_df)

df = pd.concat(df,ignore_index=True)
df.loc[df['Nuclear volume px'] == 400,'Nuclear volume'] = np.nan
df = df.sort_values(['trackID','Frame'])
df['Time'] = df['Frame'] * 10
df = df.reset_index()

#%% Collate with cell cycle annotations, smoothe, and normalize

tracking_df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)
annos = {trackID:cell for trackID,cell in tracking_df.groupby('TrackID')}
tracks = {trackID:cell for trackID,cell in df.groupby('trackID')}

annotated = {}
for trackID,t in tracks.items():

    # Truncate since didn't do whole time course yet
    if not int(trackID) in annos.keys():
        continue
    this_anno = annos[int(trackID)]
    this_anno = this_anno.rename(columns={'FRAME':'Frame'})
    t = pd.merge(t,this_anno[['Phase','name','Frame']],on='Frame')
    t = t.rename(columns={'name':'SpotID'})
    tracks[trackID] = t

    # Calculate cell age if there is a confirmed birth frame
    birth_frames = (t['Phase'] == 'Birth') | (t['Phase'] == 'Visible Birth')
    if birth_frames.sum() > 0:
        t['Age'] = t['Time'] - t[birth_frames].iloc[0]['Time']
    else:
        t['Age'] = np.nan

    # Smooth the growth curve
    Vsm,dVsm = get_interpolated_curve(t,y_field='Nuclear volume',x_field='Time',smoothing_factor=1e10)
    t['Nuclear volume (sm)'] = Vsm
    t['Growth rates (sm)'] = dVsm
    
    # Normalize H2B/Cdt/Geminin
    t['Normalized Gem intensity'] = t['Mean Gem intensity'] / t['Mean Gem intensity'].mean()
    t['Normalized H2B intensity'] = t['Mean H2B intensity'] / t['Mean H2B intensity'].mean()
    t['Normalized Cdt1 intensity'] = t['Mean Cdt1 intensity'] / t['Mean Cdt1 intensity'].mean()
    t = t.set_index('index')
    annotated[trackID] = t

_df = pd.concat(annotated,ignore_index=True)
# _df.set_index('index')

# Merge backin
df_combined = pd.merge(df,_df,how='left')
df_combined.to_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'))

for t in annotated.values():
    plt.plot(t.Age,t['Nuclear volume'])
    
    
