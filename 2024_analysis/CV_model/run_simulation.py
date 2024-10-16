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
os.chdir('/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/CV_model')
import simulation
import pickle as pkl

#%% Load empirical data

# Load growth curves from pickle
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5-full/tracked_cells/collated_manual.pkl','rb') as f:
    c5f = pkl.load(f,encoding='latin-1')
collated = c1+c2+c5+c5f

# Filter for phase-annotated cells in collated
collated_filtered = [c for c in collated if c.iloc[0]['Phase'] != '?']

# Concatenate all collated cells into dfc
dfc = pd.concat(collated_filtered)

emp_g1_cv = stats.variation(dfc[dfc['Phase'] == 'G1']['Volume (sm)'])
emp_sg2_cv = stats.variation(dfc[dfc['Phase'] == 'SG2']['Volume (sm)'])

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

def plot_growth_curves_population(pop):
    
    plt.figure()
    for cell in pop.values():
        ts = cell.ts.dropna()
        t = ts['Time']
        v = ts['Volume']
        p = ts['Phase']
        
        t_g1 = t[p =='G1']
        v_g1 = v[p =='G1']
        plt.plot(t_g1,v_g1,'b-',alpha=0.1)
        
        t_g2 = t[p =='S/G2/M']
        v_g2 = v[p =='S/G2/M']
        plt.plot(t_g2,v_g2,'r-',alpha=0.1)
        plt.xlabel('Time (h)')
        plt.ylabel('Cell volume (fL)')
        plt.legend(['G1','S/G2/M'])
        
#%% Run model from parameter files

dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/G1timer_SG2sizer_asym05_grfluct05/'
params = pd.read_csv(path.join(dirname,'params.csv'),index_col=0)

#% Run model
runs = {}
for model_name,p in params.iloc[-3:].iterrows():

    #% 1. Reset clock and initialize
    # Initialize each cell as a DataFrame at G1/S transition so we can specify Size and RB independently
    sim_clock['Current time'] = 0
    sim_clock['Current frame'] = 0
    
    # 2. Simulation steps
    population = run_model( sim_clock, p, Ncells)
    
    # 3. Collate data for analysis
    # Filter cells that have full cell cycles
    pop2analyze = {}
    for key,cell in population.items():
        if cell.generation > 7:
            pop2analyze[key] = cell
    
    # plot_growth_curves_population(population)
    runs[model_name] = pop2analyze

with open(path.join(dirname,'adders.pkl'),'wb') as f:
    pkl.dump(runs,f)

