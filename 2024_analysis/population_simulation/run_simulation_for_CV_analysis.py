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
Ncells = 200

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

def subsample_simulation(pop, new_sampling_dt):
    
    
    return pop_subsampled

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
