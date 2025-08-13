#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 00:50:50 2024

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

# from itsample import sample

random.seed(42)

def running_average(old_avg,X,N):
    I = ~np.isnan(X)
    N[I] = N[I] + 1
    new_avg = old_avg
    new_avg[I] = old_avg[I] + (X[I]-old_avg[I])/N[I]
    
    
    return new_avg,N

def pseudo_oscillator_density(x,wavelength=1/3,dc_magnitude=0.5):
    # amplitude = (1-dc_magnitude) / wavelength / (1- np.cos(1/wavelength))
    return np.sin(x/wavelength) + 0.5 + dc_magnitude

# def pseudo_oscillator_monte_carlo(Nsample,oscillator_wavelength, ):
#     generated_numbers = []
#     while len(generated_numbers) < Nsample:
#         # Generates a 'pseudo oscillating' probability via Metropolis
#         test_number = random.rand()
        
#         # normalization
#         amplitude = (1-dc_magnitude) / oscillator_wavelength / (1- np.cos(1/oscillator_wavelength))
        
#         prob_of_test = amplitude * np.sin(test_number/ oscillator_wavelength) + dc_magnitude
#         q = random.rand()
#         if test_number > q:# Monte Carlo accept
#             generated_numbers.append(test_number)
        
#     return generated_numbers

#%%

def simulate_cells(end_time,sampling_rate,Ncells, log_normal_mu,
                   gamma_sigma=0.3,V0_CV=np.sqrt(0.03), set_size = 100, white_vol_noise={'rel':.1},
                    frame_biases=None,frame_noise=None,visualize=False, behavior ='sizer',synchrony=None):
    
    t = np.arange(0,end_time,sampling_rate)
    shadow_time = np.hstack((-t[::-1],t[1:]))
    
    valid_cells = []
    shadow_cells = []
    running_avg = np.zeros(len(t))
    running_counters = np.zeros(len(t))
    
    while len(valid_cells) < Ncells:

        # Generate with equal probability either
        # 'Shadow' cells that could be 'visible' in the tissue but are born before the movie starts
        if synchrony:
            wavelength = synchrony['wavelength']
            dc_magnitude = synchrony['dc_magnitude']
            birth_time = sample(lambda x: pseudo_oscillator_density(x,wavelength/2,dc_magnitude),1,lower_bd = 0,upper_bd =1,guess=0.5)[0]
            birth_time = (birth_time - 0.5) * t.max() * 2
        else:
            birth_time = random.choice(shadow_time,1)[0]
            
        # Find the closet birth 'frame'
        birth_index = np.where( (t-birth_time) > 0)[0]
        
        # Log normal birth size
        V0 = set_size*random.lognormal(mean=0, sigma = V0_CV)
        
        #Growth rate normal
        gamma_base = np.log(2)/log_normal_mu
        gamma = gamma_base * (1+random.randn()*gamma_sigma)
        
        # Growth model
        V = V0*np.exp((t-birth_time)*gamma)

        # Determine "valid cell cycle" times
        V[t < birth_time] = np.nan
        if behavior == 'sizer':
            V[ V > 2 * set_size] = np.nan
        elif behavior == 'adder':
            V[ V > set_size + V0] = np.nan
        elif behavior == 'timer':
            print(t)
            division_time = (random.randn()*.25 + 1) * 3
            print(division_time)
            V[t - birth_time > division_time] = np.nan
    
        # Fixed white noise at every frame
        if 'rel' in white_vol_noise.keys():
            V = V * (1 + white_vol_noise['rel']*np.random.randn(len(V)))
        elif 'fixed' in white_vol_noise.keys():
            V = V + white_vol_noise['fixed']*set_size*np.random.randn(len(V))
        # Do frame-specific noise addition
        if frame_biases:
            for bad_frame,systematic_error in frame_biases.items():
                V[bad_frame] = V[bad_frame] * systematic_error
        if frame_noise:
            for bad_frame,systematic_error in frame_noise.items():
                V[bad_frame] = V[bad_frame] * (1 + random.randn()*systematic_error)
                
        if not np.all(np.isnan(V)) and np.isnan( V[-1] ):
            # Last non-Nan value
            div_frame = np.where(~np.isnan(V))[0][-1]

            # Add to tissue average tally
            running_avg,running_counters = running_average(running_avg,V,running_counters)
            
            # if observable within movie time
            if birth_time >= 0 and birth_time < t.max():
                
                V0 = V[birth_index]
                Vf = V[div_frame]
                
                valid_cells.append({'Birth age': birth_time
                                    ,'Growth rate': gamma
                                    ,'Birth size': V0[0]
                                    ,'Division size': Vf
                                    ,'Division age':t[div_frame]})
                
                if visualize:
                    plt.plot(t,V)
                    plt.xlabel('Days')
                    plt.ylabel('Cell volume')
                    
        else: # undivided but still contributes to average
            running_avg,running_counters = running_average(running_avg,V,running_counters)
            
    valid_cells = pd.DataFrame(valid_cells)
    valid_cells['Growth'] = valid_cells['Division size'] - valid_cells['Birth size']
    valid_cells['Log birth size'] = np.log(valid_cells['Birth size'])
    valid_cells['Duration'] = valid_cells['Division age'] - valid_cells['Birth age']
    valid_cells['Gamma duration'] = gamma * valid_cells['Duration']
    
    return valid_cells, running_avg, running_counters

#%% Visualize

Ncells = 100
end_time = 7
sampling_rate = 0.5
doubling = 3
t = np.arange(0,end_time,sampling_rate)
good_cells,field_avg,num_cells_in_tissue = simulate_cells(end_time, sampling_rate, Ncells, doubling, visualize=True
                                                         ,white_vol_noise={'fixed':0}
                                                         ,behavior = 'adder')
plt.plot(t,field_avg,'k--')

#%% Perfect sizers or adders - explore effect of subsampling in time


Ncells = 2000
end_time = 120
sampling_rates = [1,2,5,12,24]

noise = 0.00
size_control_slope = np.zeros(len(sampling_rates))
size_control_CI = np.zeros(len(sampling_rates))
size_duration_slope = np.zeros(len(sampling_rates))
size_duration_CI = np.zeros(len(sampling_rates))

df = pd.DataFrame()
index = 0
for behavior in ['sizer','adder']:
    for log_n_mu in [50,100]:
        for i,sampling_rate in enumerate(sampling_rates):
            index += 1
            cells,field_avg,num_cells_in_tissue = simulate_cells(
                end_time,
                sampling_rate,
                Ncells,
                log_normal_mu = log_n_mu, # 100 -> 52h, 50 -> 30h
                white_vol_noise={'fixed':noise}, visualize=False,
                frame_biases = None,
                behavior = behavior)
            
            linreg = sm.OLS(cells.dropna()['Growth'], sm.add_constant(cells.dropna()['Birth size'])).fit()
            # print(f'Slope from cortical volume = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
            size_control_slope[i] = linreg.params.values[1]
            size_control_CI[i] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
            
            linreg = sm.OLS(cells.dropna()['Duration'], sm.add_constant(cells.dropna()['Birth size'])).fit()
            size_duration_slope[i] = linreg.params.values[1]
            size_duration_CI[i] = (linreg.conf_int().values[1,:] - linreg.params.values[1])[1]
            df.loc[index,'Behavior'] = behavior
            df.loc[index,'Sampling rate'] = sampling_rate
            df.loc[index,'Average length'] = log_n_mu
            df.loc[index,'Growth slope'] = size_control_slope[i]
            df.loc[index,'Duration slope'] = size_duration_slope[i]
            df.loc[index,'Growth CI'] = size_control_CI[i]
            df.loc[index,'Duration CI'] = size_duration_CI[i]

    plt.figure(1);plt.errorbar(sampling_rates, size_control_slope,size_control_CI);plt.xlabel('Sampling rate (h)'); plt.ylabel('Size control slope - growth');
    plt.legend(['Sizer - 52h c.c.','Adder - 52h','Sizer - 30h','Adder - 30h'])
    
    plt.figure(2);plt.errorbar(sampling_rates, size_duration_slope,size_duration_CI);plt.xlabel('Sampling rate (h)'); plt.ylabel('Size control slope - duration')
    plt.legend(['Sizer - 52h c.c.','Adder - 52h','Sizer - 30h','Adder - 30h'])

df.to_excel('//Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Noise analysis/size_control_subsample.xlsx')


