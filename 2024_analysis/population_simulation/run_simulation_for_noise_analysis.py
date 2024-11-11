#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:13:59 2024

@author: xies
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:46:58 2020

@author: xies
"""

# Numerical / graphics
import pandas as pd
import numpy as np
from numpy import random
import seaborn as sb
import matplotlib.pylab as plt

# Utilities
from os import path
from glob import glob
import pickle
import copy
from tqdm import tqdm
from scipy import stats
from basicUtils import nonan_pairs, nonans
from mathUtils import cvariation_bootstrap

import os
os.chdir('/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/population_simulation')
import simulation
import pickle as pkl

#%% Helper functions + initial variables

# Set random seed
np.random.seed(42)

# Growth rate is set to 0.01 per hour, i.e. 70hr doubling rate
max_iter = 2000
dt = 0.5 # simulation step size in hours
# Total time simulated:
print(f'Total hrs simulated: {max_iter * dt / 70} generations')
Ncells = 100

# Time information
sim_clock = {}
sim_clock['Max frame'] = max_iter
sim_clock['Max time'] = max_iter * dt
sim_clock['dt'] = dt

def run_model(sim_clock, params, Ncells):
    
    # Seed initial cells asynchronously within the first 100h (~1.5 cell cycle times)
    async_birth_times = random.uniform(size=Ncells)*70
    
    initial_birth_times = dict(zip(range(Ncells),async_birth_times))
    
    population = dict()
    next_cellID = len(population)
    sim_clock['Current time'] = 0
    sim_clock['Current frame'] = 0
    
    for t in tqdm(np.arange(sim_clock['Max frame'] - 1)):
        
        # Check if any cells will be seeded this frame
        for k,v in initial_birth_times.items():
            if sim_clock['Current time'] >= v and sim_clock['Current time']-sim_clock['dt'] < v:
                # Seed cell into population
                # print(f'Seeding: {v}')
                new_seed = simulation.Cell(k,sim_clock,params)
                population[k] = new_seed
        
        # Advance time step by one
        sim_clock['Current frame'] += 1
        sim_clock['Current time'] += sim_clock['dt']
        
        newly_borns = {}
        for this_cell in population.values():
            
            # Skip cell if divided already
            if this_cell.divided:
                continue
            else:
                
                this_cell.advance_dt(sim_clock,params)
                
                if this_cell.divided:
                    
                    # Newly divided cell: make daughter cells
                    # print(f'\nCellID #{this_cell.cellID} has divided at frame {t}')
                    # Randomly draw an asymmettry
                    a = np.abs( random.randn(1)[0]*params['InhAsym']/100)
                    daughters = this_cell.divide(next_cellID, sim_clock, asymmetry=a)
                    next_cellID += 1
                    
                    newly_borns[daughters[0].cellID] = daughters[0]
                    # newly_borns[daughters[1].cellID] = daughters[1]
        
        population.update(newly_borns)
        
    return population

def subsample_population(pop,sim_clock,new_dt):
    assert( new_dt % sim_clock['dt'] == 0)
    subsampled = dict()
    for cellID,cell in pop.items():
        subsampled[cellID] = cell._subsample(sim_clock,new_dt)
    return subsampled

def plot_growth_curves_population(pop,origin='simulation'):
    
    plt.figure()
    for cell in pop.values():
        ts = cell.ts.dropna()
        
        t = ts['Time']
        v = ts['Measured volume']
        p = ts['Phase']
        if np.nansum(t) == 0:
            continue
        
        t_g1 = t[p =='G1']
        v_g1 = v[p =='G1']
        
        if origin == 'simulation':
            t0 = 0
        elif origin == 'birth':
            t0 = np.nanmin(t)
        elif origin == 'g1s':
            t0 = np.nanmax(t_g1)
            
        plt.plot(t_g1 - t0,v_g1,'b-',alpha=0.1)
        
        if len(t_g1) > 0:
            t_g2 = np.hstack((t_g1.values[-1],t[p =='S/G2/M']))
            # print(t_g2)
            v_g2 = np.hstack((v_g1.values[-1],v[p =='S/G2/M']))
        else:
            t_g2 = t[p =='S/G2/M']
            v_g2 = v[p =='S/G2/M']
        plt.plot(t_g2 - t0,v_g2,'r-',alpha=0.1)
        plt.xlabel('Time (h)')
        plt.ylabel('Cell volume (fL)')
        plt.legend(['G1','S/G2/M'])

def plot_size_control(pop):
    

    bsizes = np.array([c.birth_size_measured for c in pop.values()])
    g1sizes = np.array([c.g1s_size_measured for c in pop.values()])
    divsizes = np.array([c.div_size_measured for c in pop.values()])
    g1_growth = g1sizes - bsizes
    total_growth = divsizes - bsizes
    
    plt.subplot(2,1,2)
    plt.scatter(bsizes,g1_growth)
    plt.title(np.polyfit(bsizes,g1_growth,1)[0])
    
    plt.subplot(2,1,1)
    g1_duration = np.array([c.g1_duration for c in pop.values()])
    plt.scatter(bsizes,g1_duration)
    plt.title(np.corrcoef(bsizes,g1_duration)[0])

#%% Run model from parameter files

sim_clock['Current time'] = 0
sim_clock['Current frame'] = 0

wt_param = pd.Series({'BirthMean':339,
                      'BirthStd':61,
                      'MsmtNoise':20,
                      'GrMean':0.011,
                      'GrStd':0.0037,
                      'GrFluct':0.05,
                      'G1S_model':'sizer',
                      'SG2M_model':'timer',
                      'InhAsym':5,
                      'G1S_sizethreshold':550,
                      'G1S_th_error':19,
                      'Tsg2mMean':16,
                      'Tsg2mStd':8})

wt = run_model(sim_clock,wt_param,Ncells)
wt = {k:c for k,c in wt.items() if c.generation > 4 and c.divided}
wt_subsample = subsample_population(wt,sim_clock,12)

#%%

plot_growth_curves_population(wt_subsample,origin='birth')

plt.figure()
# plt.hist([c.total_duration for c in wt_subsample.values()])

plt.figure()
plot_size_control(wt)
plot_size_control(wt_subsample)

#%%

dko_param = pd.Series({'BirthMean':350,
                      'BirthStd':50,
                      'MsmtNoise':0,
                      'GrMean':0.01,
                      'GrStd':0.0037,
                      'GrFluct':0.05,
                      'G1S_model':'adder',
                      'SG2M_model':'timer',
                      'InhAsym':5,
                      'G1S_added_mean':200,
                      'G1S_added_std':30,
                      'Tsg2mMean':14,
                      'Tsg2mStd':6})

dko = run_model(sim_clock,dko_param,Ncells)
dko = {k:c for k,c in dko.items() if c.birth_time > 500 and c.divided}
dko_subsample = subsample_population(dko,sim_clock,12)

plot_growth_curves_population(dko_subsample,origin='birth')

plt.figure()
plot_size_control(dko)
plt.figure()
plot_size_control(dko_subsample)


#%%

dko_param = pd.Series({'BirthMean':350,
                      'BirthStd':50,
                      'MsmtNoise':0,
                      'GrMean':0.01,
                      'GrStd':0.0037,
                      'GrFluct':0.05,
                      'G1S_model':'adder',
                      'SG2M_model':'timer',
                      'InhAsym':5,
                      'G1S_added_mean':50,
                      'G1S_added_std':30,
                      'Tsg2mMean':14,
                      'Tsg2mStd':6})

dko = run_model(sim_clock,dko_param,Ncells)
dko = {k:c for k,c in dko.items() if c.birth_time > 500 and c.divided}
dko_subsample = subsample_population(dko,sim_clock,12)

plot_growth_curves_population(dko_subsample,origin='birth')

plt.figure()
plot_size_control(dko)
plt.figure()
plot_size_control(dko_subsample)







