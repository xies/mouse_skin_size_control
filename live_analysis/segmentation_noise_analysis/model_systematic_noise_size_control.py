#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:58:49 2023

@author: xies
"""

import numpy as np
from numpy import random
import statsmodels.api as sm
from basicUtils import *
import pandas as pd
from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm
import seaborn as sb

def simulate_cells(t,Ncells, doubling_time, gamma_sigma,V0_CV,set_size = 1,overall_noise_sigma=.1,
                    frame_biases=None,frame_noise=None,visualize=False, behavior ='sizer'):
    
    valid_cells = []
    
    while len(valid_cells) < Ncells:
        
        birth_frame = random.randint(16)
        V0 = set_size*random.lognormal(mean=0, sigma = V0_CV)
        
        gamma_base = np.log(2)/doubling_time
        gamma = gamma_base * (1+random.randn()*gamma_sigma)
        
        V = V0*np.exp((t-birth_frame/2)*gamma)
        V = V + overall_noise_sigma*np.random.randn(len(V))
        V[t < birth_frame/2] = np.nan
        if behavior == 'sizer':
            DIVIDED = V[-1] > 2*set_size
        elif behavior == 'adder':
            DIVIDED = V[-1] > set_size+V0
        
        if DIVIDED:
            
            if behavior == 'sizer':
                div_frame = np.where(V>2*set_size)[0][0]-1
                V[V>2] = np.nan
            elif behavior == 'adder':
                div_frame = np.where(V>set_size+V0)[0][0]-1
                V[V>1+V0] = np.nan
            
            Vf = V[div_frame]
            # Do correction
            if frame_biases:
                for bad_frame,systematic_error in frame_biases.items():
                    V[bad_frame] = V[bad_frame] * systematic_error
            if frame_noise:
                for bad_frame,systematic_error in frame_noise.items():
                    V[bad_frame] = V[bad_frame] * (1 + random.randn()*systematic_error)
                    
            valid_cells.append({'Birth frame': birth_frame
                                ,'Growth rate': gamma
                                ,'Birth size': V[birth_frame]
                                ,'Division size': Vf
                                ,'Duration':(~np.isnan(V)).sum()
                                ,'Division frame':div_frame})
            if visualize:
                plt.plot(t,V)
                plt.xlabel('Days')
                plt.ylabel('Cell volume')
                
    valid_cells = pd.DataFrame(valid_cells)
    
    valid_cells['Growth'] = valid_cells['Division size'] - valid_cells['Birth size']
    valid_cells['Log birth size'] = np.log(valid_cells['Birth size'])
    valid_cells['Gamma duration'] = gamma * valid_cells['Duration']
    
    return valid_cells

#%% Visualize

Ncells = 200
set_size = 100
gamma_sigma = 0.3
t = np.arange(0,8,.5)
V0_CV = np.sqrt(0.03) # sqrt(0.03) is around 20% CV

plt.figure()
good_cells = simulate_cells(t, Ncells, 3, gamma_sigma, V0_CV, visualize=True, behavior = 'adder')

bad_frames = {3: 0.8, 8: 1.2, 12:1.2}
plt.figure()
bad_cells = simulate_cells(t, Ncells, 3, gamma_sigma, V0_CV, frame_biases = bad_frames,visualize=True,behavior = 'adder')

plt.figure()
sb.regplot(good_cells,x='Birth size',y='Growth')
sb.regplot(bad_cells,x='Birth size',y='Growth')

#%% Perfect sizers - explore effect of sampling rate

Ncells = 200
gamma_sigma = 0.2
t = np.arange(0,8,.5)
V0_CV = np.sqrt(0.03) # sqrt(0.03) is around 20% CV

doubling_times = np.arange(1,10) # in days
avg_duration = np.zeros(len(doubling_times))
size_control_slope = np.zeros(len(doubling_times))
size_control_CI = np.zeros(len(doubling_times))
size_duration_slope = np.zeros(len(doubling_times))
size_duration_CI = np.zeros(len(doubling_times))

for i,doubling in enumerate(doubling_times):

    size_control = simulate_sizers(t, Ncells, doubling, gamma_sigma, V0_CV,visualize=False)

    # plt.figure()
    # sb.regplot(size_control,x='Log birth size',y='Gamma duration')

    avg_duration[i] = size_control['Duration'].mean()
    
    linreg = sm.OLS(size_control.dropna()['Growth'], sm.add_constant(size_control.dropna()['Birth size'])).fit()
    # print(f'Slope from cortical volume = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
    size_control_slope[i] = linreg.params.values[1]
    size_control_CI[i] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
    
    linreg = sm.OLS(size_control.dropna()['Gamma duration'], sm.add_constant(size_control.dropna()['Log birth size'])).fit()
    size_duration_slope[i] = linreg.params.values[1]
    size_duration_CI[i] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]

plt.figure();plt.errorbar(avg_duration/2, size_control_slope,size_control_CI);plt.xlabel('Avg length of cell cycle (days)'); plt.ylabel('Size control slope - growth')
plt.figure();plt.errorbar(avg_duration/2, size_duration_slope,size_duration_CI);plt.xlabel('Avg length of cell cycle (days)'); plt.ylabel('Size control slope - duration')

#%% Systematic biases

Niter = 100
Ncells = 600
gamma_sigma = 0.2
t = np.arange(0,8,.5)
V0_CV = np.sqrt(0.03)

size_control_slope = np.zeros((9,Niter))
size_control_CI = np.zeros((9,Niter))
size_duration_slope = np.zeros((9,Niter))
size_duration_CI = np.zeros((9,Niter))

for i, num_bad_frames in tqdm(enumerate(np.arange(0,9))):

    for j in range(Niter):
        
        bad_frames = {f : 1+0.2*random.randn() for f in sample_without_replacement(len(t),num_bad_frames)}
        new_cells = simulate_cells(t, Ncells, 3.5, gamma_sigma, V0_CV, frame_biases= bad_frames, visualize=False,behavior='adder')
        
        linreg = sm.OLS(new_cells.dropna()['Growth'], sm.add_constant(new_cells.dropna()['Birth size'])).fit()
        size_control_slope[i,j] = linreg.params.values[1]
        size_control_CI[i,j] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
        
        linreg = sm.OLS(new_cells.dropna()['Gamma duration'], sm.add_constant(new_cells.dropna()['Log birth size'])).fit()
        size_duration_slope[i,j] = linreg.params.values[1]
        size_duration_CI[i,j] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
        
plt.figure();plt.errorbar(np.arange(0,9), size_control_slope.mean(axis=1),size_control_CI.mean(axis=1));plt.xlabel('# of systematically biased frames'); plt.ylabel('Size control slope - growth')
plt.figure();plt.errorbar(np.arange(0,9), size_duration_slope.mean(axis=1),size_duration_CI.mean(axis=1));plt.xlabel('# of systematically biased frames'); plt.ylabel('Size control slope - duration')

plt.figure()
sb.regplot(valid_cells,x='Birth size',y='Growth')
sb.regplot(new_cells,x='Birth size',y='Growth',scatter_kws={'alpha':0.5})
plt.legend(['Original','w/Systematic bias'])

plt.figure()
sb.regplot(valid_cells,x='Log birth size',y='Gamma duration',y_jitter=0.05)
sb.regplot(new_cells,x='Log birth size',y='Gamma duration',y_jitter=0.05,scatter_kws={'alpha':0.5})
plt.legend(['Original','w/Systematic bias'])

#%% Heteroskedastic

Ncells = 600
gamma_sigma = 0.3
t = np.arange(0,8,.5)
V0_CV = np.sqrt(0.03)


size_control_slope = np.zeros((9,Niter))
size_control_CI = np.zeros((9,Niter))
size_duration_slope = np.zeros((9,Niter))
size_duration_CI = np.zeros((9,Niter))

good_cells = simulate_cells(t, Ncells, 3.5, gamma_sigma, V0_CV, visualize=False,behavior='adder')

for i, num_bad_frames in tqdm(enumerate(np.arange(0,9))):
    
    for j in range(Niter):
        
        bad_frames = {f : 0.2*random.randn() for f in sample_without_replacement(len(t),num_bad_frames)}
        new_cells = simulate_cells(t, Ncells, 3.5, gamma_sigma, V0_CV, frame_noise = bad_frames, visualize=False,behavior='adder')

        new_cells['Growth'] = new_cells['Division size'] - new_cells['Birth size']
        
        linreg = sm.OLS(new_cells.dropna()['Growth'], sm.add_constant(new_cells.dropna()['Birth size'])).fit()
        # print(f'Slope from cortical volume = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
        size_control_slope[i,j] = linreg.params.values[1]
        size_control_CI[i,j] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
        
        linreg = sm.OLS(new_cells.dropna()['Log duration'], sm.add_constant(new_cells.dropna()['Birth size'])).fit()
        size_duration_slope[i,j] = linreg.params.values[1]
        size_duration_CI[i,j] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
        
    

plt.figure();plt.errorbar(np.arange(0,9), size_control_slope.mean(axis=1),size_control_CI.mean(axis=1));plt.xlabel('# of large noise frames'); plt.ylabel('Size control slope - growth')
plt.figure();plt.errorbar(np.arange(0,9), size_duration_slope.mean(axis=1),size_duration_CI.mean(axis=1));plt.xlabel('# of large noise frames'); plt.ylabel('Size control slope - duration')

plt.figure()
sb.regplot(good_cells,x='Birth size',y='Growth')
sb.regplot(new_cells,x='Birth size',y='Growth',scatter_kws={'alpha':0.5})
plt.legend(['Original','w/farame noise'])

plt.figure()
sb.regplot(good_cells,x='Log birth size',y='Gamma duration',y_jitter=0.05)
sb.regplot(new_cells,x='Log birth size',y='Gamma duration',y_jitter=0.05,scatter_kws={'alpha':0.5})
plt.legend(['Original','w/frame noise'])
