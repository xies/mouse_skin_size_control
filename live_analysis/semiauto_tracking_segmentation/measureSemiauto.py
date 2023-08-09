#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:26:02 2023

@author: xies
"""

import numpy as np
import pandas as pd

from skimage import io
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def measure_track_timeseries_from_segmentations(name,pathdict,metadata):
    '''
    Given a dictionary of filenames to: 'Segmentation':tracked labels
                                        'H2B': H2b intensity image
                                        'FUCCI': FUCCI intensity iamge
                                        'Frame averages':CSV file of per-frame nuclear sizes (in pixel)
    And dictionary of metadata: 'um_per_px'
                                'Region': name of region
                                'Mouse' : mouse name
                                'Pair' : littermate pair
                                'Genotype'
                                'Dirname'
    
    Return a list of 'tracks' where each track is the time-series of a single cell
    
    '''
    
    print('Loading segmentation...')
    manual_segs = io.imread(pathdict['Segmentation'])
    
    frame_averages = pd.read_csv(pathdict['Frame averages'],index_col=0)
    frame_averages = frame_averages.groupby('Frame')['area'].mean()
    print('Loading H2B...')
    G = io.imread(pathdict['H2B'])
    print('Loading FUCCI...')
    R = io.imread(pathdict['FUCCI'])
    
    
    # parse metadata
    dx = metadata['um_per_px']
    name = metadata['Region']
    mouse = metadata['Mouse']
    pair = metadata['Pair']
    genotype = metadata['Genotype']
    dirname = metadata['Dirname']
    mode = metadata['Mode']
    
    # Parse the cell cycle annotation only to figure out where to stop
    anno = pd.read_excel(pathdict['Cell cycle annotations'],usecols=range(5),index_col=0,sheet_name=mode)
    trackIDs = anno.index
    
    tracks = []
    print('Working on each trackID...')
    for trackID in tqdm(trackIDs):
        
        track = []
        
        mask = manual_segs == trackID
        # mask_uncorrected = filtered_segs == trackID
        frames_with_this_track = np.where(np.any(np.any(np.any(mask,axis=1),axis=1), axis=1))[0]
        
        for frame in frames_with_this_track:
            
            # Measurements from segmentation/ label iamge
            this_frame = mask[frame,...]
            # this_frame_threshed = this_frame & G_th[frame,...]
            
            # props = measure.regionprops(this_frame*1)
            Z,Y,X = np.where(this_frame)
            Z = Z.mean();Y = Y.mean();X = X.mean()
            volume = this_frame.sum()
            # thresholded_volume = this_frame_threshed.sum() * dx**2
            
            if volume == 1000:
                volume = np.nan
            volume = volume 
            
            # Measurement from intensity image(s)
            h2b_this_frame = G[frame,...]
            h2b_mean = h2b_this_frame[this_frame].mean()
            
            fucci_this_frame = R[frame,...]
            fucci_mean = fucci_this_frame[this_frame].mean()
            
            track.append(pd.DataFrame({'Frame':frame,'X':X,'Y':Y,'Z':Z,'Volume pixels':volume
                                       ,'Volume': volume * dx**2
                                       # ,'Volume thresh': thresholded_volume
                                       ,'Volume normal': volume / (frame_averages.loc[frame]) # work only with pixels no need to calibrate
                                       ,'H2b mean':h2b_mean
                                       ,'FUCCI mean':fucci_mean},index=[frame]))
        
        if len(track) > 0:
            
            track = pd.concat(track)
            track['CellID'] = trackID
            if 'Time stamps' in metadata.keys():
                track['Age'] = metadata['Time stamps'][track['Frame'].values.astype(int)]
            else:
                track['Age'] = (track['Frame'] - track.iloc[0]['Frame'])*12
            track['Region'] = name
            track['Genotype'] = genotype
            track['Mouse'] = mouse
            track['Pair'] = pair
            track['um_per_px'] = dx
            track['Directory'] = dirname
            track['Mode'] = mode
            
            tracks.append(track)

        
    print('Detecting missing frames and filling-in...')
    
    #Go back and detect the missing frames and fill-in with NaNs
    for i,track in enumerate(tracks):
        
        frames = track['Frame']
        skipped_frames = frames[:-1][np.diff(frames) > 1] # the 'before skip' frame
        for f in skipped_frames:
            
            missing_frame = f+1
            track = pd.concat([track,pd.Series({'Frame':missing_frame,'X':np.nan,'Y':np.nan,'Z':np.nan,'Volume':np.nan
                                 ,'CellID' : track.iloc[0].CellID, 'Age': (missing_frame - track.iloc[0]['Frame']) * 12
                                 ,'Region':name,'Genotype':genotype
                                 })],ignore_index=True)
            track = track.sort_values('Frame').reset_index(drop=True)
    
    print('Smoothing and calculating growth rates...')
    
    for i,track in enumerate(tracks):
        
        track['Volume interp'],spl = smooth_growth_curve(track,y='Volume')
        track['Volume normal interp'],spl_norm = smooth_growth_curve(track,y='Volume normal')

        if not spl == None:
        
            track['Growth rate'] = spl.derivative(1)(track['Age'])
            track['Growth rate normal'] = spl_norm.derivative(1)(track['Age'])
        
        # track['Growth rate back'] = np.diff(track['Volume'])
        # track['Growth rate interp'] = track['Spline interp'].iloc[0].derivateve(1)(track['Age'])

        tracks[i] = track


    print(f'Done with {name}')
    
    return tracks
    
def cell_cycle_annotate(tracks,pathdict,metadata):
    '''
    Given a list of 'tracks' (see measure_track_timeseries_from_segmentations)
    and a pathdict with 'Cell cycle annotations' pointing to the cell cycle annotation Excel file
    return tracks with cell cycle phases annotated
    '''
    
    mode = metadata['Mode']
    
    print('Cell cycle annotating...')
    anno = pd.read_excel(pathdict['Cell cycle annotations'],usecols=range(5),index_col=0,sheet_name=mode)
    for track in tracks:
        
        track['Phase'] = 'NA'
        track['Birth frame'] = np.nan
        track['Division frame'] = np.nan
        track['S phase entry frame'] = np.nan
        track['Mitosis'] = 'No'
        track['Annotated'] = False
        
        if track.iloc[0].CellID in anno.index:
            this_anno = anno.loc[track.iloc[0].CellID]
            track['Annotated'] = True
            
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
    
    return tracks

def collate_timeseries_into_cell_centric_table(tracks,metadata):
    
    
    dx = metadata['um_per_px']
    name = metadata['Region']
    mouse = metadata['Mouse']
    pair = metadata['Pair']
    genotype = metadata['Genotype']
    dirname = metadata['Dirname']
    mode = metadata['Mode']
    time = metadata['Time stamps']
    
    print(f'Generating cell table for {name}')
    
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
        if (not np.isnan(birth_frame)) and (birth_frame in track['Frame'].values):
                
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
        
        if (not np.isnan(div_frame)) and (not div_frame in track['Frame']):
            age = time[int(div_frame)] - track.iloc[0]['Age']
            track = pd.concat([track,pd.DataFrame({'Frame':div_frame,'X':np.nan,'Y':np.nan,'Z':np.nan
                                            ,'Volume':np.nan
                                            ,'Volume thresh':np.nan
                                            # ,'Volume normal':np.nan
                                            ,'CellID' : track.iloc[0].CellID,
                                            'Age': age},index=[int(div_frame)]
                                               )],ignore_index=True)
            track = track.sort_values('Frame').reset_index(drop=True)
            
        if (not np.isnan(s_frame)) and (not s_frame in track['Frame']):
            s_age = time[int(s_frame)] - track.iloc[0]['Age']
            track = pd.concat([track,pd.DataFrame({'Frame':s_frame,'X':np.nan,'Y':np.nan,'Z':np.nan
                                            ,'Volume':np.nan
                                            ,'Volume thresh':np.nan
                                            #, 'Volume normal':np.nan
                                            ,'CellID' : track.iloc[0].CellID,
                                            'Age': s_age}
                                                ,index=[int(s_frame)])
                                         ],ignore_index=True)
        
        # Division
        if not np.isnan(div_frame):
            # print(div_frame)
            # print(track['CellID'])
            div_size = track[track['Frame'] == div_frame]['Volume'].values[0]
            div_size_normal = track[track['Frame'] == div_frame]['Volume normal'].values[0]
            div_size_interp = track[track['Frame'] == div_frame]['Volume interp'].values[0]
            div_size_normal_interp = track[track['Frame'] == div_frame]['Volume normal interp'].values[0]
            total_length = track[track['Frame'] == div_frame]['Age'].values[0]
        else:
            div_size = np.nan
        
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
                    ,'um_per_px':dx
                    ,'Region':name
                    ,'Genotype':genotype
                    ,'Mouse':mouse
                    ,'Pair':pair
                    ,'Dirname':dirname
                    ,'Mode':mode
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

    df = pd.DataFrame(df)
    
    df['G1 growth'] = df['S phase entry size'] - df['Birth size']
    df['Total growth'] = df['Division size'] - df['Birth size']
    df['G1 growth normal'] = df['S phase entry size normal'] - df['Birth size normal']
    df['Total growth normal'] = df['Division size normal'] - df['Birth size normal']
    df['G1 growth interp'] = df['S phase entry size interp'] - df['Birth size interp']
    df['Total growth '] = df['Division size interp'] - df['Birth size interp']
    df['G1 growth normal interp'] = df['S phase entry size normal interp'] - df['Birth size normal interp']
    df['Total growth normal interp'] = df['Division size normal interp'] - df['Birth size normal interp']
    
    return df, tracks

def recalibrate_pixel_size(tracks,cell_table,new_dx):
    
    old_dx = tracks[0].iloc[0]['um_per_px']
    rescaling_factor = new_dx**2 / old_dx**2
    
    for i,t in enumerate(tracks):
        
        t['Volume'] = t['Volume'] * rescaling_factor
        t['Volume interp'] = t['Volume interp'] * rescaling_factor
        t['um_per_px'] = new_dx
        tracks[i] = t
        
    cell_table['Birth size'] = cell_table['Birth size'] * rescaling_factor
    cell_table['S phase entry size'] = cell_table['S phase entry size'] * rescaling_factor
    cell_table['Division size'] = cell_table['Division size'] * rescaling_factor
    
    cell_table['Birth size interp'] = cell_table['Birth size interp'] * rescaling_factor
    cell_table['S phase entry size interp'] = cell_table['S phase entry size interp'] * rescaling_factor
    cell_table['Division size interp'] = cell_table['Division size interp'] * rescaling_factor
    
    cell_table['G1 growth'] = cell_table['S phase entry size'] - cell_table['Birth size']
    cell_table['G1 growth interp'] = cell_table['S phase entry size interp'] - cell_table['Birth size interp']
    cell_table['Total growth'] = cell_table['Division size'] - cell_table['Birth size']
    cell_table['Total growth interp'] = cell_table['Division size interp'] - cell_table['Birth size interp']
    
    cell_table['um_per_px'] = new_dx
    
    return tracks,cell_table
    
    
    
def smooth_growth_curve(cf,x='Age',y='Volume',smoothing_factor=1e10):

    X = cf[x]
    Y = cf[y]
    
    I = (~np.isnan(X)) * (~np.isnan(Y))
        
    # Won't smooth 3 pts or fewer (cubic spline)
    if len(X[I]) < 4:
        Yhat = cf[y].values
        spl = None
        
    else:

        # Spline smooth
        spl = UnivariateSpline(X[I], Y[I], k=3, s=smoothing_factor)
        Yhat = spl(X)
        
    return Yhat, spl
    


    
    
    
