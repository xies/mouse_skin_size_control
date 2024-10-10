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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2

filt_seg = io.imread(path.join(dirname,'manual_cellcycle_annotations/filtered_segs.tif'))
T = filt_seg.shape[0]

#%%

pl = pv.Plotter()
df = []
for t in tqdm(range(T)):
    
    all_labels = io.imread(path.join(dirname,f'manual_segmentation/man_Channel0-T{t+1:04d}.tif'))
    props = measure.regionprops(all_labels)
    _df = pd.DataFrame(index=range(len(props)),columns=['cellID', 'Nuclear volume'
                                                        ,'Axial moment','Axial angle'
                                                        ,'Planar moment 1'
                                                        ,'Planar moment 2'
                                                        ,'Planar angle'
                                                        ,'Z','Y','X'
                                                        ,'Surface area'])
    
    
    for i,p in enumerate(props):
        _df['Frame'] = t
        _df.loc[i,'cellID'] = p['label']
        _df.loc[i,'Nuclear volume px'] = p['area']
        _df.loc[i,'Nuclear volume'] = p['area'] * dx**2 * dz
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        z,y,x = p['centroid']
        _df.loc[i,'Z'] = z*dz
        _df.loc[i,'Y'] = y*dx
        _df.loc[i,'X'] = x*dx
        _df.loc[i,'Z - px'] = z
        _df.loc[i,'Y - px'] = y
        _df.loc[i,'X - px'] = x
        if p['area'] > 400:
        
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
            
            _df.loc[i,'Surface area'] = mesh.area
            _df.loc[i,'Mesh volume'] = mesh.volume
    
    # Load the cell tracking and merge the trackID via centroid matching
    _tracked = pd.DataFrame(measure.regionprops_table(filt_seg[t,...],properties=['centroid','label']))
    _tracked = _tracked.rename(columns={'label':'trackID',
                                        'centroid-0':'Z - px','centroid-1':'Y - px','centroid-2':"X - px"})
    _df = pd.merge(_df,_tracked,on=['X - px','Y - px','Z - px'],how='left')
    
    df.append(_df)
    
df = pd.concat(df,ignore_index=True)
df.loc[df['Nuclear volume px'] == 400,'Nuclear volume'] = np.nan
df = df.sort_values(['trackID','Frame'])
df['Time'] = df['Frame'] * 10
df = df.reset_index()

#%% Collate with cell cycle annotations and then smooth

tracking_df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)
annos = {trackID:cell for trackID,cell in tracking_df.groupby('TrackID')}
tracks = {trackID:cell for trackID,cell in df.groupby('trackID')}

for trackID,t in tracks.items():
    
    # Truncate since didn't do whole time course yet
    this_anno = annos[trackID]
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
    t = t.set_index('index')
    
_df = pd.concat(tracks,ignore_index=True)
_df.set_index('index')

# Merge backin
df_combined = pd.merge(df,_df,how='left')
df_combined.to_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'))

for t in tracks.values():
    plt.plot(t.Age,t['Nuclear volume (sm)'])

#%%

size_summary = df.groupby(['trackID','Phase']).mean()['Nuclear volume (sm)']
size_summary = size_summary.reset_index()

size_summary = pd.pivot(size_summary,index='trackID',columns=['Phase'])
size_summary.columns = size_summary.columns.droplevel(0)
size_summary['G1 growth'] = size_summary['G1S'] - size_summary['Visible birth']


age_summary = df.groupby(['trackID','Phase']).min()['Age']
age_summary = age_summary.reset_index()
age_summary = pd.pivot(age_summary,index='trackID',columns=['Phase'])
age_summary.columns = age_summary.columns.droplevel(0)
age_summary['G1 length'] = age_summary['G1S'] - age_summary['Birth']

summary = pd.merge(age_summary['G1 length'],size_summary, on='trackID')



plt.scatter(summary['Visible birth'],summary['G1 length'])
plt.scatter(summary['Visible birth'],summary['G1 growth'])


#%%

df_g1s = df[ (df['Phase'] == 'Visible birth') | (df['Phase'] == 'G1') | (df['Phase'] == 'G1S')]
df_g1s['Nuclear volume (sm)'] = df_g1s['Nuclear volume (sm)'].astype(float)
df_g1s['G1S_logistic'] = df['Phase'] == 'G1S'

sb.regplot(df_g1s,x='Nuclear volume (sm)',y='G1S_logistic',logistic=True, y_jitter=.1, scatter_kws={'alpha':0.1})




