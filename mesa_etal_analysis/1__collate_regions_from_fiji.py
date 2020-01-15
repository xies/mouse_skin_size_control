#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:01:43 2019

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
import os.path as path
import re
from scipy import stats
from scipy.interpolate import UnivariateSpline
import pickle as pkl


dx = 0.25
regions = {'/Users/xies/box/Mouse/Skin/Mesa et al//W-R1/tracked_cells/':'M1R1',
            '/Users/xies/box/Mouse/Skin/Mesa et al/W-R2/tracked_cells/':'M1R2',
            '/Users/xies/box/Mouse/Skin/Mesa et al/W-R5/tracked_cells/':'M2R5',
            '/Users/xies/box/Mouse/Skin/Mesa et al/W-R5-full/tracked_cells/':'M2R5'}

for regiondir in regions.iterkeys():
    # Grab single-frame data into a dataframe
    raw_df = pd.DataFrame()
    frames = []
    cIDs = []
    vols = []
    fucci = []
    daughter = []
    nuclei = []
    
    filelist = glob(path.join(regiondir,'*/*.txt'))
    for fullname in filelist:
        subdir,f = path.split(fullname)
        # Skip the log.txt or skipped.txt file
        if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt' or f == 'mitosis_in_frame.txt':
            continue
        fn, extension = path.splitext(f)
        if extension == '.txt':
            fn, channel = path.splitext(fn)
    #            print fullname
            # Measure everything on FUCCI channel first
            if channel == '.fucci':
                subdir = path.split(subdir)[1]
                # Grab cellID from subdir name
                cIDs.append( np.int(path.split(subdir)[1]) )
                
                # Add segmented area to get volume (um3)
                # Add total FUCCI signal to dataframe
                cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
                vols.append(cell['Area'].sum())
                fucci.append(cell['Mean'].mean())
                
                # Check if main lineage or daughter cells
                framename = path.split(fn)[1]
                match = re.search('(\D)$',framename)
                
                # Main cell linage
                if match == None:
                    # Grab the frame # from filename
                    frame = f.split('.')[0]
                    frame = np.int(frame[1:])
                    frames.append(frame)
                    daughter.append('None')
                else:
                    daughter_name = match.group(0)
                    frame = f.split('.')[0]
                    frame = np.int(frame[1:-1])
                    frames.append(frame)
                    daughter.append(daughter_name)
            elif channel == '.h2b':
                cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
                nuclei.append(cell['IntDen'].sum().astype(np.float) * dx**2)
        
    raw_df['Frame'] = frames
    raw_df['CellID'] = cIDs
    raw_df['Volume'] = vols
    raw_df['Nucleus'] = nuclei
    raw_df['G1'] = fucci
    raw_df['Daughter'] = daughter
    raw_df['NC ratio'] = np.array(vols) / np.array(nuclei)
    
    # Load hand-annotated G1/S transition frame
    g1transitions = pd.read_csv(path.join(regiondir,'g1_frame.txt'),',')
    
    # Collate cell-centric list-of-dataslices
    ucellIDs = np.unique( raw_df['CellID'] )
    Ncells = len(ucellIDs)
    collated = []
    for c in ucellIDs:
        # Grab data from the raw dataframe
        this_cell = raw_df[raw_df['CellID'] == c].sort_values(by='Frame').copy()
        this_cell['Region'] = regions[regiondir]
        this_cell = this_cell.reset_index()
        
        # Annotate age in hours
        this_cell['Age'] = (this_cell['Frame'] - this_cell['Frame'].min()) * 12
        
        # Annotate cell cycle of parent cell
        transition_frame = g1transitions[g1transitions.CellID == this_cell.CellID[0]].iloc[0].Frame
        if transition_frame == '?':
            this_cell['Phase'] = '?'
        else:
            this_cell['Phase'] = 'SG2'
            iloc = np.where(this_cell.Frame == np.int(transition_frame))[0][0]
            this_cell.loc[0:iloc,'Phase'] = 'G1'
        # Annotate cell cycle of daughter cell
        this_cell.loc[this_cell['Daughter'] != 'None','Phase'] = 'Daughter G1'
        
        # Store spline-fits
        spline_fit = get_interpolated_curve(this_cell)
        this_cell['Volume (sm)'] = spline_fit
        
        # Store pointwise growth rates (smoothed)
        gr,gr_sm = get_growth_rate(this_cell)
        this_cell['Growth rate'] = gr / 12.0
        this_cell['Growth rate (sm)'] = gr_sm / 12.0
        
        # Append to master list
        collated.append(this_cell)
    
    # Load mitosis frame
    mitosis_in_frame = pd.read_csv(path.join(regiondir,'mitosis_in_frame.txt'),',')
    # Annotate mitosis as 'M' in 'Phase'
    if len(mitosis_in_frame) > 1:
        for i,mitosis in mitosis_in_frame.iterrows():
            c = collated[np.where(ucellIDs == mitosis.CellID)[0][0]]
            c.loc[c.Frame == mitosis.mitosis_frame,'Phase'] = 'M'
    
    
    ##### Export growth traces in CSV ######
    pd.concat(collated).to_csv(path.join(regiondir,'growth_curves.csv'),
                            index=False)
    
    f = open(path.join(regiondir,'collated_manual.pkl'),'w')
    pkl.dump(collated,f)
    
    # Cell-centric data frame construction
    # Collapse into single cell v. measurement DataFrame
    Tcycle = np.zeros(Ncells)
    Bsize = np.zeros(Ncells)
    Bframe = np.zeros(Ncells)
    DivSize = np.zeros(Ncells)
    G1duration = np.zeros(Ncells)
    G1size = np.zeros(Ncells)
    G1size_interp = np.zeros(Ncells)
    cIDs = np.zeros(Ncells)
    daughterSizes = np.zeros((2,Ncells))
    daughterGR = np.zeros(Ncells) * np.nan
    finalGR = np.zeros(Ncells) * np.nan
    expGR = np.zeros(Ncells)
    for i,cf in enumerate(collated):
        # Break out the daughter cells
        d = cf[cf['Daughter'] != 'None']
        d = d.iloc[0:2]
        c = cf[cf['Daughter'] == 'None']
        
        cIDs[i] = c['CellID'].iloc[0]
        Bsize[i] = c['Volume'].iloc[0]
        Bframe[i] = c['Frame'].iloc[0]
        DivSize[i] = c['Volume'][len(c)-1]
        Tcycle[i] = len(c) * 12
        # Find manual G1 annotation
        thisg1frame = g1transitions[g1transitions['CellID'] == c['CellID'].iloc[0]]['Frame'].values[0]
        if thisg1frame == '?':
            G1duration[i] = np.nan
            G1size[i] = np.nan
        else:
            thisg1frame = np.int(thisg1frame)
            G1duration[i] = (thisg1frame - c.iloc[0]['Frame'] + 1) * 12
            G1size[i] = c[c['Frame'] == thisg1frame]['Volume']
            G1size_interp[i] = c[c['Frame'] == thisg1frame]['Volume (sm)']

        # Annotate daughter cell data
        if len(d) > 0:
            daughterSizes[0,i] = d.iloc[0]['Volume']
            daughterSizes[1,i] = d.iloc[1]['Volume']
            daughterGR[i] = np.sum(d['Growth rate'])
            if np.isnan(d.iloc[0]['Growth rate']):
                daughterGR[i] = np.nan
        else:
            daughterSizes[:,i] = np.nan
        
        # Store exponential growth rate from fit
        expGR[i] = get_exponential_growth_rate(c)
        # Store final Growth rate
        finalGR[i] = c.iloc[-1]['Growth rate']
        
    # Construct dataframe with primary data
    df = pd.DataFrame()
    df['CellID'] = cIDs
    df['Birth frame'] = Bframe
    df['Cycle length'] = Tcycle
    df['G1 length'] = G1duration
    df['G1 volume'] = G1size
    df['G1 volume interpolated'] = G1size_interp
    df['Birth volume'] = Bsize
    df['Division volume'] = DivSize
    df['Daughter a volume'] = daughterSizes[0,:]
    df['Daughter b volume'] = daughterSizes[1,:]
    df['Exponential growth rate'] = expGR
    df['Final growth rate'] = finalGR
    df['Combined daughter growth rate'] = daughterGR
    
    # Derive data
    df['Daughter total volume'] = df['Daughter a volume'] + df['Daughter b volume']
    df['Daughter ratio'] = np.vstack((df['Daughter a volume'], df['Daughter b volume'])).min(axis=0) / \
                            np.vstack((df['Daughter a volume'], df['Daughter b volume'])).max(axis=0)
    df['Division volume interpolated'] = (df['Daughter total volume'] + df['Division volume'])/2
    df['Total growth'] = df['Division volume'] - df['Birth volume']
    df['SG2 length'] = df['Cycle length'] - df['G1 length']
    df['G1 grown'] = df['G1 volume'] - df['Birth volume']
    df['G1 grown interpolated'] = df['G1 volume interpolated'] - df['Birth volume']
    df['SG2 grown'] = df['Total growth'] - df['G1 grown']
    df['Fold grown'] = df['Division volume'] / df['Birth volume']
    df['Total growth interpolated'] = df['Division volume interpolated'] - df['Birth volume']
    df['SG2 grown interpolated'] = df['Division volume interpolated'] - df['G1 volume interpolated']
    
    # Put in the mitosis annotation
    df['Mitosis'] = np.in1d(df.CellID,mitosis_in_frame)

    # Filter out cells with no phase information        
    df_nans = df
    df = df[~np.isnan(df['G1 grown'])]    
    df['Region'] = regions[regiondir]
    
    #Pickle the dataframe
    df.to_pickle(path.join(regiondir,'dataframe.pkl'))
    
    print 'Done with:', regiondir
    
    
    
def get_interpolated_curve(c,smoothing_factor=1e5):

    # Get rid of daughter cells
    cf = c[c['Daughter'] == 'None']
    if len(cf) < 4:
        yhat = cf.Volume.values
        
    else:
        t = np.array(range(0,len(cf))) * 12
        v = cf.Volume.values
        # Spline smooth
        spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
        yhat = spl(t)

    # if cell had daughter points bit
    ndaughters = (c['Phase'] == 'Daughter G1').sum()
    if ndaughters > 0:
        nan_padding = np.ones((1,ndaughters)) * np.nan
        yhat = np.append( yhat, nan_padding )
    
    return yhat
    

def get_growth_rate(c):
    
    # Get rid of daughter cells
    cf = c[c['Daughter'] == 'None']

    v = cf['Volume'].values
    v_sm = cf['Volume (sm)'].values
    Tb = backward_difference(len(v))
    gr = np.dot(Tb,v)
    Tb = backward_difference(len(v_sm))
    gr_sm = np.dot(Tb,v)
    gr[0] = np.nan
    gr_sm[0] = np.nan

    # Calculate daughter growth rate if available
    ndaughters = (c['Phase'] == 'Daughter G1').sum()
    if ndaughters == 2:
        # Can't calculate daughter G1 growth rate
        gr = np.append( gr,[np.nan,np.nan] )
        gr_sm = np.append( gr_sm,[np.nan,np.nan] )
    
    elif ndaughters == 4:
        # Separate daughter a + daughter b
        daughter_a = c[c['Daughter'] == 'a']
        daughter_b = c[c['Daughter'] == 'b']
        
        gra = daughter_a.iloc[-1].Volume - daughter_a.iloc[0].Volume
        grb = daughter_b.iloc[-1].Volume - daughter_b.iloc[0].Volume
        
        gr = np.append(gr, [gra,grb,np.nan,np.nan])
        gr_sm = np.append(gr_sm, [gra,grb,np.nan,np.nan])
    
    return gr,gr_sm

def get_exponential_growth_rate(c):
    # Return the exponential fit growth rate parameter
    from scipy import optimize
    exp_model = lambda x,p1,p2,p3 : p1 * np.exp(p2 * x) + p3

    t = c.Age.values
    v = c.Volume.values
    # Construct initial guess for growth rate
    
    try:
        # Nonlinear regression
        b = optimize.curve_fit(exp_model,t,v,p0 = [v[0],1,v.min()],
                                 bounds = [ [0,0,v.min()],
                                            [v.max(),np.inf,v.max()]])
        return b[0][1]
    
    except:
        return np.nan
        print 'Fitting failed for ', c.iloc[0].CellID

