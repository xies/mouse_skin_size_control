#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure, exposure, util, segmentation
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl
from twophotonUtils import smooth_growth_curve

dirnames = {}
# dirnames['WT_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
dirnames['WT_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
# dirnames['WT_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R1'

# dirnames['RBKO_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
# dirnames['RBKO_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R1'
# dirnames['RBKO_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R2'

dx = {}
dx['WT_R1'] = 0.2920097/1.5
dx['WT_R2'] = 0.2920097/1.5
dx['WT_R3'] = 0.206814922817745/1.5
dx['RBKO_R1'] = 0.2920097/1.5
dx['RBKO_R2'] = 0.2920097/1.5
dx['RBKO_R3'] = 0.206814922817745/1.5
dx['RBKO_R4'] = 0.206814922817745/1.5

RECALCULATE = True

def plot_cell_volume(track,x='Frame',y='Volume'):
    t = track[x]
    y = track[y]
    if track.iloc[0]['Mitosis']:
        t = t[:-1]
        y = y[:-1]
    plt.plot(t,y)
    
limit = {'WT_R1':51,'WT_R2':103,'WT_R3':66,'RBKO_R1':60,'RBKO_R2':52,'RBKO_R3':85, 'RBKO_R4':53}

mode = 'manual'

#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)

for name,dirname in dirnames.items():
    
    print(f'---- Working on {name} ----')
    
    genotype = name.split(' ')[0]
    
    #% Re-construct tracks with manually fixed tracking/segmentation
    if RECALCULATE:
        
        # filtered_segs = io.imread(path.join(dirname,'manual_tracking/filtered_segmentation.tif'))
        # manual_segs = io.imread(path.join(dirname,'manual_tracking/manual_tracking_clahe.tif')) 
        manual_segs = io.imread(path.join(dirname,f'manual_tracking/{mode}_clahe.tif'))
        frame_averages = pd.read_csv(path.join(dirname,'high_fucci_avg_size.csv'))
        frame_averages = frame_averages.groupby('Frame').mean()['area']
        # for t in tqdm(range(17)):
        #     manual_segs[t,...] = segmentation.expand_labels(manual_segs[t,...],distance=1)    
        
        G = io.imread(path.join(dirname,'master_stack/G.tif'))
        R = io.imread(path.join(dirname,'master_stack/R.tif'))
        # G_th = io.imread(path.join(dirname,'master_stack/G_clahe.tif'))
        print('Loaded images')
        
        trackIDs = np.unique(manual_segs)
        
        tracks = []
        for trackID in tqdm(trackIDs[1:limit[name]]):
            
            track = []
            
            mask = manual_segs == trackID
            # mask_uncorrected = filtered_segs == trackID
            frames_with_this_track = np.where(np.any(np.any(np.any(mask,axis=1),axis=1), axis=1))[0]
            
            for frame in frames_with_this_track:
                
                # Measurements from segmentation/ label iamge
                this_frame = mask[frame,...]
                # this_frame_threshed = this_frame & G_th[frame,...]
                
                props = measure.regionprops(this_frame*1)
                Z,Y,X = np.where(this_frame)
                Z = Z.mean();Y = Y.mean();X = X.mean()
                volume = this_frame.sum()
                # thresholded_volume = this_frame_threshed.sum() * dx**2
                
                if volume == 1000:
                    volume = np.nan
                    thresholded_volume = np.nan
                volume = volume * dx[name]**2
                
                # Measurement from intensity image(s)
                h2b_this_frame = G[frame,...]
                h2b_mean = h2b_this_frame[this_frame].mean()
                
                fucci_this_frame = R[frame,...]
                fucci_mean = fucci_this_frame[this_frame].mean()
                
                track.append(pd.DataFrame({'Frame':frame,'X':X,'Y':Y,'Z':Z,'Volume':volume
                                           # ,'Volume thresh': thresholded_volume
                                           ,'Volume normal': volume / (frame_averages.loc[frame] * dx[name]**2)
                                           # ,'H2b mean':h2b_mean
                                           ,'FUCCI mean':fucci_mean},index=[frame]))
            
            track = pd.concat(track)
            track['CellID'] = trackID
            track['Age'] = (track['Frame'] - track.iloc[0]['Frame'])*12
            track['Region'] = name
            track['Genotype'] = genotype
            
            tracks.append(track)

            print(f'Done with CellID {trackID}')
        
        #Go back and detect the missing frames and fill-in with NaNs
        for i,track in enumerate(tracks):
            
            frames = track['Frame']
            skipped_frames = frames[:-1][np.diff(frames) > 1] # the 'before skip' frame
            for f in skipped_frames:
                
                missing_frame = f+1
                track = track.append(pd.Series({'Frame':missing_frame,'X':np.nan,'Y':np.nan,'Z':np.nan,'Volume':np.nan
                                     ,'CellID' : track.iloc[0].CellID, 'Age': (missing_frame - track.iloc[0]['Frame']) * 12
                                     ,'Region':name,'Genotype':genotype
                                     }),ignore_index=True)
                track = track.sort_values('Frame').reset_index(drop=True)
                tracks[i] = track
        
        # Save to the manual folder    
        with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'wb') as file:
            pkl.dump(tracks,file)
    
    #% Load volume annotations
    with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'rb') as file:
        tracks = pkl.load(file)
    
    # Load excel annotations of cell cycle
    # Also smooth volume curve from existing raw curves
    filename = path.join(dirname,f'{name}_cell_cycle_annotations.xlsx')
    anno = pd.read_excel(filename,usecols=range(5),index_col=0)
    for track in tracks:
        
        track['Phase'] = 'NA'
        track['Birth frame'] = np.nan
        track['Division frame'] = np.nan
        track['S phase entry frame'] = np.nan
        track['Mitosis'] = 'No'
        
        if track.iloc[0].CellID in anno.index:
            this_anno = anno.loc[track.iloc[0].CellID]
            
            track['Birth frame'] = this_anno.Birth
            track['Division frame'] = this_anno.Division
            track['S phase entry frame'] = this_anno['S phase entry']
            
            if not np.isnan(this_anno['S phase entry']):
                track.loc[track['Frame'] < this_anno['S phase entry'],'Phase'] = 'G1'
                track.loc[track['Frame'] >= this_anno['S phase entry'],'Phase'] = 'S'
            
            track.loc[track['Frame'] == this_anno.Birth,'Phase'] = 'Birth'
            
            # if not np.isnan(this_anno['Division']):
            #     track.loc[track['Frame'] == this_anno.Division,'Phase'] = 'Division'
            
            track['Mitosis'] = this_anno['Mitosis?'] == 'Yes'
            if track.iloc[0]['Mitosis']:
                track.loc[track['Frame'] == this_anno.Division,'Volume'] = np.nan
        
        track['Volume interp'] = smooth_growth_curve(track,y='Volume')
        track['Volume normal interp'] = smooth_growth_curve(track,y='Volume normal')
    
    # Save to the manual folder    
    with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'wb') as file:
        pkl.dump(tracks,file)

    
    with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'rb') as file:
        tracks = pkl.load(file)
    
    # Construct the cell-centric metadata dataframe
    df = []
    for track in tracks:
        
        birth_size = np.nan
        div_size = np.nan
        s_size = np.nan
        birth_size_normal = np.nan
        div_size_normal = np.nan
        s_size_normal = np.nan
        birth_size_interp = np.nan
        birth_size_normal_interp = np.nan
        s_size_interp = np.nan
        s_size_normal_interp = np.nan
        div_size_interp = np.nan
        div_size_normal_interp = np.nan
        
        g1_length = np.nan
        total_length = np.nan
        
        # Birth
        birth_frame = track.iloc[0]['Birth frame']
        if not np.isnan(birth_frame):
            birth_size = track[track['Frame'] == birth_frame]['Volume'].values[0]
            birth_size_normal = track[track['Frame'] == birth_frame]['Volume normal'].values[0]
            birth_size_interp = track[track['Frame'] == birth_frame]['Volume interp'].values[0]
            birth_size_normal_interp = track[track['Frame'] == birth_frame]['Volume normal interp'].values[0]
            
        else:
            birth_frame = np.nan
        
        div_frame = track.iloc[0]['Division frame']
        s_frame = track.iloc[0]['S phase entry frame']
        
        # it's possible that the div/s frame isn't in the segmentation frame
        # because that frame has bad quality -> fill in with NA
        if not div_frame in track['Frame']:
            track = track.append(pd.Series({'Frame':div_frame,'X':np.nan,'Y':np.nan,'Z':np.nan
                                            ,'Volume':np.nan
                                            ,'Volume thresh':np.nan
                                            # ,'Volume normal':np.nan
                                 ,'CellID' : track.iloc[0].CellID, 'Age': (div_frame - track.iloc[0]['Frame']) * 12
                                 }),ignore_index=True)
            track = track.sort_values('Frame').reset_index(drop=True)
            
        if not s_frame in track['Frame']:
            track = track.append(pd.Series({'Frame':s_frame,'X':np.nan,'Y':np.nan,'Z':np.nan
                                            ,'Volume':np.nan
                                            ,'Volume thresh':np.nan
                                            #, 'Volume normal':np.nan
                                 ,'CellID' : track.iloc[0].CellID, 'Age': (s_frame - track.iloc[0]['Frame']) * 12
                                 }),ignore_index=True)
        
        # Division
        if not np.isnan(div_frame):
            div_size = track[track['Frame'] == div_frame]['Volume'].values[0]
            div_size_normal = track[track['Frame'] == div_frame]['Volume normal'].values[0]
            div_size_interp = track[track['Frame'] == div_frame]['Volume interp'].values[0]
            div_size_normal_interp = track[track['Frame'] == div_frame]['Volume normal interp'].values[0]
            total_length = track[track['Frame'] == div_frame]['Age'].values[0]
        else:
            div_frame = np.nan
        
        # Delete mitotic volumes
        if track.iloc[0].Mitosis:
            div_size = np.nan
        
        # G1/S
        if not np.isnan(s_frame):
            s_size = track[track['Frame'] == s_frame]['Volume'].values[0]
            s_size_normal = track[track['Frame'] == s_frame]['Volume normal'].values[0]
            s_size_interp = track[track['Frame'] == s_frame]['Volume interp'].values[0]
            s_size_normal_interp = track[track['Frame'] == s_frame]['Volume normal interp'].values[0]
            g1_length = track[track['Frame'] == s_frame]['Age'].values[0]
        else:
            s_frame = np.nan
        
        df.append({'CellID':track.iloc[0].CellID
                   ,'Region':name
                    ,'Genotype':genotype
                    ,'Birth frame': birth_frame
                    ,'S phase frame': s_frame
                    ,'Division frame':div_frame
                    ,'Birth size': birth_size
                    ,'Birth size normal': birth_size_normal
                    ,'Birth size interp': birth_size_interp
                    ,'Birth size normal interp': birth_size_normal_interp
                    ,'S phase entry size':s_size
                    ,'S phase entry size normal': s_size_normal
                    ,'S phase entry size interp':s_size_interp
                    ,'S phase entry size normal interp': s_size_normal_interp
                    ,'Division size': div_size
                    ,'Division size normal': div_size_normal
                    ,'Division size interp': div_size_interp
                    ,'Division size normal interp': div_size_normal_interp
                    ,'G1 length':g1_length
                    ,'SG2 length':total_length - g1_length
                    ,'Total length':total_length
                    # ,'G1 growth':s_size - birth_size
                    # ,'Total growth':div_size - birth_size
                    # ,'G1 growth normal':s_size_normal - birth_size_normal
                    # ,'Total growth normal':div_size_normal - birth_size_normal
                    })                    

    # Save to the manual folder    
    with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'wb') as file:
        pkl.dump(tracks,file)

    df = pd.DataFrame(df)
    
    df['G1 growth'] = df['S phase entry size'] - df['Birth size']
    df['Total growth'] = df['Division size'] - df['Birth size']
    df['G1 growth normal'] = df['S phase entry size normal'] - df['Birth size normal']
    df['Total growth normal'] = df['Division size normal'] - df['Birth size normal']
    df['G1 growth interp'] = df['S phase entry size interp'] - df['Birth size interp']
    df['Total growth '] = df['Division size interp'] - df['Birth size interp']
    df['G1 growth normal interp'] = df['S phase entry size normal interp'] - df['Birth size normal interp']
    df['Total growth normal interp'] = df['Division size normal interp'] - df['Birth size normal interp']
    
    
    df.to_csv(path.join(dirname,f'{name}_manual_tracking/dataframe_{mode}.csv'))


#%% basic plotting



# IDs2plot = np.arange(20,30)

# for corrID in IDs2plot:
#     plt.plot(tracks[corrID]['Frame'],tracks[corrID]['Volume'])
    
    
    


  