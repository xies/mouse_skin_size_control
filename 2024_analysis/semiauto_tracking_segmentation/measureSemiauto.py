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
from scipy.optimize import curve_fit

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
    
    if pathdict['Frame averages'] != '':
        frame_averages = pd.read_csv(pathdict['Frame averages'],index_col=0)
        frame_averages = frame_averages.groupby('Frame')['area'].mean()
        NORMALIZE = True
    else:
        NORMALIZE = False
        
    print('Loading H2B...')
    G = io.imread(pathdict['H2B'])
    print('Loading FUCCI...')
    R = io.imread(pathdict['FUCCI'])
    
    
    # parse metadata
    dx = metadata['um_per_px']
    dz = metadata['um_per_slice']
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
    for trackID in trackIDs:
        
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
            print(this_frame.shape)
            print(h2b_this_frame.shape)
            h2b_mean = h2b_this_frame[this_frame].mean()
            
            fucci_this_frame = R[frame,...]
            fucci_mean = fucci_this_frame[this_frame].mean()
            if NORMALIZE:
                vol_norm = volume / (frame_averages.loc[frame]) # work only with pixels no need to calibrate
            else:
                vol_norm = np.nan
            track.append(pd.DataFrame({'Frame':frame,'X':X,'Y':Y,'Z':Z,'Volume pixels':volume
                                       ,'Volume': volume * dx**2 *dz
                                       ,'Volume normal': vol_norm
                                       ,'H2b mean':h2b_mean
                                       ,'FUCCI mean':fucci_mean},index=[frame]))
            
        
        if len(track) > 0:
            
            track = pd.concat(track)
            track['CellID'] = trackID
            if 'Time stamps' in metadata.keys():
                track['Age'] = metadata['Time stamps'][(track['Frame'] - track.iloc[0]['Frame']).astype(int)]
                if track.iloc[0].Age > 0:
                    what
                
            else:
                what
                track['Age'] = (track['Frame'] - track.iloc[0]['Frame'])*12
            track['Region'] = name
            track['Genotype'] = genotype
            track['Mouse'] = mouse
            track['Pair'] = pair
            track['um_per_px'] = dx
            track['um_per_slice'] = dz
            track['Directory'] = dirname
            track['Mode'] = mode
            
            tracks.append(track)

        
    print('Detecting missing frames and filling-in...')
    
    #Go back and detect the missing frames and fill-in with NaNs
    for i,track in enumerate(tracks):
        
        frames = track['Frame']
        skipped_frames = frames.iloc[:-1][np.diff(frames) > 1] # the 'before skip' frame
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
        track['Growth rate'] = np.nan
        track['Growth rate normal'] = np.nan
        track['Specific GR normal'] = np.nan
        track['Specific GR'] = np.nan
         
        
        if not spl == None:
        
            track['Growth rate'] = spl.derivative(1)(track['Age'])
            track['Specific GR'] = track['Growth rate'] / track['Volume interp']
            if NORMALIZE:
                track['Growth rate normal'] = spl_norm.derivative(1)(track['Age'])
                track['Specific GR normal'] = track['Growth rate normal'] / track['Volume normal interp']
        
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
    anno = pd.read_excel(pathdict['Cell cycle annotations'],usecols=range(6),index_col=0,sheet_name=mode)
    if 'Generation' in anno.columns:
        GENERATION = True
        print('GENERATION')
    else:
        GENERATION = False
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
            
            if GENERATION:
                track['Generation'] = this_anno['Generation']
            
    for t in tracks:
        t['Time to G1/S'] = t['Frame'] - t['S phase entry frame']
    
    return tracks


def annotate_ablation_distance(tracks,metadata):

    ablations = metadata['Ablated cell coords']
    for i,t in enumerate(tracks):
        for j,row in t.iterrows():
            dx = row['X'] - ablations['X']
            dy = row['Y'] - ablations['Y']
            D = np.sqrt(dx**2 + dy**2)
            t['Distance to ablated cell'] = D.min()
        tracks[i] = t
        
    return tracks
    
    # for i in range(Nablations):
    #     abl = ablations.iloc[i]
    #     dx = df['X'] - abl['X']
    #     dy = df['Y'] - abl['Y']
        
    #     D[:,i] = dx**2 + dy**2
    # return D.min(axis=1)


def collate_timeseries_into_cell_centric_table(tracks,metadata):
    
    
    dx = metadata['um_per_px']
    dz = metadata['um_per_slice']
    name = metadata['Region']
    mouse = metadata['Mouse']
    pair = metadata['Pair']
    genotype = metadata['Genotype']
    dirname = metadata['Dirname']
    mode = metadata['Mode']
    time = metadata['Time stamps']
    
    print(f'Generating cell table for {name}_{mode}')
    print('--')
    
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
        
        # If birth frame is not NAN, then parse the birth size
        birth_frame = track.iloc[0]['Birth frame']
        if (not np.isnan(birth_frame)) and (birth_frame in track['Frame'].values):
                
                birth_size = track[track['Frame'] == birth_frame]['Volume'].values[0]
                birth_size_normal = track[track['Frame'] == birth_frame]['Volume normal'].values[0]
                birth_size_interp = track[track['Frame'] == birth_frame]['Volume interp'].values[0]
                birth_size_normal_interp = track[track['Frame'] == birth_frame]['Volume normal interp'].values[0]
            
        
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
        
        # If division isn't in the 
        if not np.isnan(div_frame):
            # print(div_frame)
            # print(track['CellID'])
            div_size = track[track['Frame'] == div_frame]['Volume'].values[0]
            div_size_normal = track[track['Frame'] == div_frame]['Volume normal'].values[0]
            div_size_interp = track[track['Frame'] == div_frame]['Volume interp'].values[0]
            div_size_normal_interp = track[track['Frame'] == div_frame]['Volume normal interp'].values[0]
            total_length = track[track['Frame'] == div_frame]['Age'].values[0]
        
        # Delete mitotic volumes
        if track.iloc[0].Mitosis:
            div_size = np.nan
        
        # Grab cell size at S phase entry; s_frame must NOT be the first frame (i.e. two frames to observe transition point)
        if not np.isnan(s_frame) and s_frame > 0:
            
            if s_frame - 1 in track.Frame.values:
                #NB: Use average size between the last FUCCI-high and first FUCCI low frame
                first_low_size = track[track['Frame'] == s_frame]['Volume'].values[0]
                last_high_size = track[track['Frame'] == s_frame-1]['Volume'].values[0]
                s_size = (first_low_size + last_high_size)/2
                
                first_low_size = track[track['Frame'] == s_frame]['Volume normal'].values[0]
                last_high_size = track[track['Frame'] == s_frame-1]['Volume normal'].values[0]
                s_size_normal = (first_low_size + last_high_size)/2
                
                first_low_size = track[track['Frame'] == s_frame]['Volume interp'].values[0]
                last_high_size = track[track['Frame'] == s_frame-1]['Volume interp'].values[0]
                s_size_interp = (first_low_size + last_high_size)/2
                
                first_low_size = track[track['Frame'] == s_frame]['Volume normal interp'].values[0]
                last_high_size = track[track['Frame'] == s_frame-1]['Volume normal interp'].values[0]
                s_size_normal_interp = (first_low_size + last_high_size)/2
                
                first_low_age = track[track['Frame'] == s_frame]['Age'].values[0]
                last_high_age = track[track['Frame'] == s_frame-1]['Age'].values[0]
                g1_length = (first_low_age+last_high_age)/2
            
            ## This uses the exact value @ first observed FUCCI-low frame
            # s_size = track[track['Frame'] == s_frame]['Volume'].values[0]
            # s_size_normal = track[track['Frame'] == s_frame]['Volume normal'].values[0]
            # s_size_interp = track[track['Frame'] == s_frame]['Volume interp'].values[0]
            # s_size_normal_interp = track[track['Frame'] == s_frame]['Volume normal interp'].values[0]
            # g1_length = track[track['Frame'] == s_frame]['Age'].values[0]

        # Exponential fit parameters
        
        params,_ = exponential_fit(track,x='Age',y='Volume normal')
        V0 = params[0]
        gamma = params[1]

        # If annotated, add generation #
        if 'Generation' in track.columns:
            generation = track.iloc[0]['Generation']
        else:
            generation = np.nan

        df.append({'CellID':track.iloc[0].CellID
                    ,'um_per_px':dx
                    ,'um_per_slice':dz
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
                    ,'Exponential growth rate':gamma
                    ,'Exponential initial':V0
                    ,'Generation':generation
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
    df['Total growth interp'] = df['Division size interp'] - df['Birth size interp']
    
    df['G1 growth normal interp'] = df['S phase entry size normal interp'] - df['Birth size normal interp']
    df['Total growth normal interp'] = df['Division size normal interp'] - df['Birth size normal interp']
    
    df['S entry size'] = df['Birth size'] + df['G1 growth']
    df['Log birth size'] = np.log(df['Birth size'])
    df['Fold grown'] = df['Division size'] / df['Birth size']
    df['SG2 growth'] = df['Total growth'] - df['G1 growth']
    
    return df, tracks

def recalibrate_pixel_size(tracks,cell_table,new_dx=None,new_dz=None):
    
    old_dx = tracks[0].iloc[0]['um_per_px']
    old_dz = tracks[0].iloc[0]['um_per_slice']
    
    if new_dx == None:
        new_dx = old_dx
    if new_dz == None:
        new_dz = old_dz
    
    rescaling_factor = new_dx**2 / old_dx**2
    rescaling_factor = new_dz/old_dz
    
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
    cell_table['um_per_slice'] = new_dz
    
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
    
def exponential(t, V0, gamma):
    return np.exp(gamma*t)*V0

def exponential_fit(cf,x='Age',y='Volume normal'):

    X = cf[x]
    Y = cf[y]
    
    I = (~np.isnan(X)) * (~np.isnan(Y))
    X = X[I]
    Y = Y[I]
    
    if len(X) < 4:
        params = np.array([np.nan, np.nan])
        pCOV = np.nan
    else:
        init_guess = [Y.mean(),0.01]
        params, pCOV = curve_fit(exponential,X,Y,p0=init_guess)
        
    return params, pCOV
        

