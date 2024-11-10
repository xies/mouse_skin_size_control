#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:45:04 2020

@author: xies
"""

import numpy as np
from numpy import random
import pandas as pd
import scipy as sp
from copy import deepcopy
from  mathUtils import estimate_log_normal_parameters

class Cell():
    
    def __init__(self, cellID, sim_clock, params=None, mother=None, inheritance=0.5):
        '''
        Cell for simulating cell growth/division coupling

        Parameters
        ----------
        cellID : int
            Unique cell ID
        sim_clock : dict
            Dictionary containing the time information of simulation
        params : pd.DataFrame
            empirical parameters extracted from DFB dataset
        mother : Cell, optional
            If inheriting properties from a mother. The default is None.
        inheritance : TYPE, optional
            fraction of inheritance from mother. The default is 0.5.

        Returns
        -------
        Cell object.
        
        
        Definitions
        -----------
        
        Volume = cell volume

        '''
        
        self.cellID = cellID
        
        # ts -- time series
        self.ts = pd.DataFrame( columns = ['Time','Volume','Phase','Measured volume'],
                            index=pd.RangeIndex( sim_clock['Max frame'] ))
        
        # empirical parameters
        self.exp_growth_rate = np.nan
        self.g1s_size_threshold = np.nan
        self.sg2m_size_threshold = np.nan
        self.g1s_adder = np.nan
        self.sg2m_adder = np.nan
        
        # Sizes
        self.birth_size = np.nan
        self.birth_size_measured = np.nan
        self.div_size= np.nan
        self.div_size_measured = np.nan
        self.g1s_size = np.nan
        self.g1s_size_measured = np.nan
        
        # Timing
        self.birth_time = np.nan
        self.div_time = np.nan
        self.g1s_time = np.nan
        
        # Frame
        self.birth_frame = np.nan
        self.div_frame = np.nan
        self.g1s_frame = np.nan
        
        # Durations
        self.g1_duration = np.nan
        self.sg2_duration = np.nan
        self.total_duration = np.nan
        
        self.params = None
        
        # Flag for if cell is has already divided
        self.divided = False
        
        # Check if a mother cell was passed in during construction
        if mother == None:
            
            # @todo: create a helper function
            # If Mother is not provided, must provide empirical parameters to initialize de novo
            # Will initialize a cell at birth
            assert(params is not None)
            # Create cell de novo using empirical paramters
            # Random log-normal birth sizes
            mu,sigma = estimate_log_normal_parameters(params['BirthMean'],params.loc['BirthStd'])
            birth_vol = random.lognormal(mean =mu, sigma=sigma)
            # Add measurement noise
            V_noise = random.randn(1)[0]*params['MsmtNoise']
            # Random normal exp growth rates
            gr = random.randn(1)[0]*params['GrStd'] + params['GrMean']
            gr = max(0,gr)
            
            
            init_cell = {'Time':sim_clock['Current time']
                                ,'Volume':birth_vol
                                ,'Measured volume':birth_vol+V_noise
                                ,'Phase':'G1'}
            self.parentID = np.nan
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            self.birth_size = birth_vol
            self.birth_size_measured = birth_vol+V_noise
            self.exp_growth_rate = gr
            
            # Set up the G1S decision function
            if params['G1S_model'].lower() == 'sizer':
                theta = random.randn(1)[0]*params['G1S_th_error'] + params['G1S_sizethreshold']
                theta = max(theta,300)
                self.g1s_size_threshold = theta
            if params['G1S_model'].lower() == 'adder':
                DV = random.randn(1)[0]*params['G1S_added_std'] * params['G1S_added_mean']
                DV = max(50,DV)
                self.g1s_adder = DV
            if params['G1S_model'].lower() == 'timer':
                Tg1 = random.randn(1)[0]*(params['Tg1Std']) + params['Tg1Mean']
                # Impose minimum of 1h
                Tg1 = max(Tg1,5)
                self.g1_duration = Tg1
            
            # Set up the division decision function
            if params['SG2M_model'].lower() == 'sizer':
                theta = random.randn(1)[0]*params['SG2M_th_error'] + params['SG2M_sizethreshold']
                theta = max(theta,500)
                self.sg2m_size_threshold = theta
            if params['SG2M_model'].lower() == 'adder':
                DV = random.randn(1)[0]*params['SG2M_added_std'] * params['SG2M_added_mean']
                DV = max(50,DV)
                self.sg2m_adder = DV
            if params['SG2M_model'].lower() == 'timer':
                # Pre-determine SG2M duration to avoid doing the random processes math
                Tsg2m = random.randn(1)[0]*(params['Tsg2mStd']) + params['Tsg2mMean']
                # Impose minimum of 1h
                Tsg2m = max(Tsg2m,5)
                self.sg2m_duration = Tsg2m
            
            self.params = params
            self.generation = 0
            
        else:
            
            # Create daughter cell via symmetric division
            #NB: the "halving" is taken care of in @divide function
            self.params = mother.params
            params = self.params
            init_vol = mother.div_size * inheritance
            
            V_noise = random.randn(1)[0]*params['MsmtNoise']
            init_cell = {'Time':sim_clock['Current time'],'Measured volume':init_vol+V_noise,
                                   'Volume':init_vol,'Phase':'G1'}
            self.parentID = mother.cellID
            self.birth_size = init_vol
            self.birth_size_measured = init_vol+V_noise
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            
            if params['G1S_model'].lower() == 'sizer':
                # @todo: Is this inherited?
                theta = mother.g1s_size_threshold
                # theta = random.randn(1)[0]*params['G1S_th_error'] + params['G1S_sizethreshold']
                # theta = max(theta,300)
                self.g1s_size_threshold = theta
            if params['G1S_model'].lower() == 'adder':
                DV = random.randn(1)[0]*params['G1S_added_std'] * params['G1S_added_mean']
                DV = max(50,DV)
                self.g1s_adder = DV
            elif params['G1S_model'].lower() == 'timer':
                Tg1 = random.randn(1)[0]*(params['Tg1Std']) + params['Tg1Mean']
                # Impose minimum of 1h
                Tg1 = max(Tg1,5)
                self.g1_duration = Tg1
            
            #Pick new growth rate
            self.exp_growth_rate = random.randn(1)[0]*params['GrStd'] + params['GrMean']
            self.generation = mother.generation + 1
            
            # Pre-determine SG2M duration to avoid doing the random processes math
            if params['SG2M_model'].lower() == 'sizer':
                theta = random.randn(1)[0]*params['SG2M_th_error'] + params['SG2M_sizethreshold']
                theta = max(theta,500)
                self.sg2m_size_threshold = theta
            if params['SG2M_model'].lower() == 'adder':
                DV = random.randn(1)[0]*params['SG2M_added_std'] * params['SG2M_added_mean']
                DV = max(50,DV)
                self.sg2m_adder = DV
            if params['SG2M_model'].lower() == 'timer':
                Tsg2m = random.randn(1)[0]*(params['Tsg2mStd']) + params['Tsg2mMean']
                # Impose minimum of 1h
                Tsg2m = max(Tsg2m,5)
                self.sg2m_duration = Tsg2m
        
        self.ts.iloc[ sim_clock['Current frame'],:] = init_cell
        
    
    def divide( self, cellID_beginning, sim_clock, asymmetry = 0.0):
        '''
        Divides the current mother cell into two daughters.

        Parameters
        ----------
        cellID_beginning : int
            the cellID of first daughter, +1 will be second daughter
        sim_clock : dict
            Dictionary containing the time information of simulation
        asymmetry : float, optional
            The difference between larger daughter + smaller daughter normalized by mother.
            sym = (D_L - D_s) / M
            The default is 0.0.

        Returns
        -------
        daughter_a : Cell
            Larger daughter
        daughter_b : Cell
            Smaller daughter

        '''
        # current_frame = sim_clock['Current frame']
        self.g1s_duration = self.g1s_time - self.birth_time
        self.sg2m_duration = self.div_time - self.g1s_time
        self.total_duration = self.div_time - self.birth_time
        
        # Calculate respective inheritance fractions 
        assert(asymmetry < 1.0)

        inh_a = (asymmetry + 1.) / 2
        inh_b = 1 - (asymmetry + 1.) / 2
            
        daughter_a = Cell(cellID_beginning, sim_clock, mother=self, inheritance=inh_a)
        daughter_b = Cell(cellID_beginning+1, sim_clock, mother=self, inheritance=inh_b)
        
        return (daughter_a, daughter_b)

# --- SIMULATION METHODS ----

    def advance_dt(self,clock,params):
        '''
        
        Parameters
        ----------
        clock : dict
            Dictionary containing current simulation time information, keyed by frame
        params : pd.Dataframe
            Empirical parameters
            
        Currently only uses Euler updates
             
        @todo: use Runge-Kutta for stability -- stick to Euler for now, num stability is no problem

        '''
        
        assert(self.divided == False) # sanity check that there is no division yet
        
        prev_frame = clock['Current frame'] - 1
        prev_values = self.ts.iloc[prev_frame] # pd.Series
        
        # Initialize current cell's values as pd.Series from prev. time point
        current_values = prev_values.copy()
        current_values['Time'] += clock['dt']
        
        # Grow cell in volume following parameter sheet
        dV = self.volume_growth(clock,params)
        
        final_V = current_values['Volume'] + dV
        final_V = max(final_V,0)
        current_values['Volume'] = final_V
        
        #Add measurement noise
        V_noise = random.randn(1)[0]*params['MsmtNoise']
        final_V = current_values['Volume']+V_noise
        final_V = max(0,final_V)
        current_values['Measured volume'] = final_V
        
        # Check for G1/S transition
        if prev_values['Phase'] == 'G1' and self.g1s_transition(clock,params):
            current_values['Phase'] = 'S/G2/M'
            self.g1s_time = current_values['Time']
            self.g1s_size = current_values['Volume']
            self.g1s_size_measured = current_values['Measured volume']
            self.g1s_frame = clock['Current frame']
            
        # Check for division
        if prev_values['Phase'] == 'S/G2/M' and self.decide2divide(clock,params):
            
            current_values['Volume'] = np.nan
            self.div_time = prev_values['Time']
            self.div_size = prev_values['Volume']
            self.div_size_measured = prev_values['Measured volume']
            self.div_frame = prev_frame
            self.divided = True
        
        # Put the current values into the timeseries
        self.ts.iloc[prev_frame + 1,:] = current_values
        
    
    def volume_growth(self, sim_clock, params):
        frame = sim_clock['Current frame'] - 1
        # Simple exponential growth
        cell = self.ts.iloc[frame]
        gr = self.exp_growth_rate
        # Add fluctuation in gr
        gr = gr*(1 + random.randn(1)[0]*params['GrFluct'])
        # Rewriter new growth rate
        self.exp_growth_rate = gr
        dV = cell['Volume'] * gr # Euler update
        
        return dV * sim_clock['dt']
        
    def g1s_transition(self,sim_clock,params):
        
        # Always 'work' based on prev_frame information
        frame = sim_clock['Current frame'] -1
        cell = self.ts.iloc[frame]
        
        assert(cell['Phase'] == 'G1')
        
        if params['G1S_model'].lower() == 'sizer':
            return cell['Volume'] > self.g1s_size_threshold
        
        elif params['G1S_model'].lower() == 'adder':
            added_since_birth = cell['Volume'] - self.birth_size
            return added_since_birth > self.g1s_adder
        
        elif params['G1S_model'].lower() == 'timer':
            time_of_birth = self.birth_time
            time_since_birth = cell['Time'] - time_of_birth
            return (time_since_birth > self.g1_duration)
        else:
            error
    
    def decide2divide(self,sim_clock,params):
        # Always 'work' based on prev_frame information
        frame = sim_clock['Current frame'] - 1
        # Decide to divide if S/G2/M duration has lasted a certain time duration (following empirical parameter sheet)
        
        cell = self.ts.iloc[frame]
        # Only S/G2/M cells divide
        assert(cell['Phase'] == 'S/G2/M')
        
        if params['SG2M_model'].lower() == 'sizer':
            return cell['Volume'] > self.sg2m_size_threshold
        
        elif params['SG2M_model'].lower() == 'adder':
            added_since_g1s = cell['Volume'] - self.g1s_size
            return added_since_g1s > self.sg2m_adder
        
        elif params['SG2M_model'].lower() == 'timer':
            # Calculate time since g1s
            time_of_g1s = self.g1s_time
            time_since_g1s = sim_clock['Current time'] - time_of_g1s
            return time_since_g1s > self.sg2m_duration
        else:
            # Todo: error
            return None

    def _subsample(self,sim_clock,sub_sample_dt):
        assert( sub_sample_dt % sim_clock['dt'] == 0)
        
        decimation_factor = int(sub_sample_dt // sim_clock['dt'])
        
        subsampled = deepcopy(self)
        
        # decimate timeseries
        decimated_ts = self.ts[::decimation_factor]
        subsampled.ts = decimated_ts
        
        # recalculate cell cycle positions
        birth_idx = decimated_ts['Time'].argmin()
        
        # Birth
        subsampled.birth_size = decimated_ts.iloc[birth_idx]['Volume']
        subsampled.birth_size_measured = decimated_ts.iloc[birth_idx]['Measured volume']
        subsampled.birth_time = decimated_ts.iloc[birth_idx]['Time']
        subsampled.birth_frame = birth_idx
        
        if 'S/G2/M' in decimated_ts['Phase'].unique():
            g1s_idx = np.where(decimated_ts['Phase'] == 'S/G2/M')[0][0]
            
            # G1/S
            subsampled.g1s_size = decimated_ts.iloc[g1s_idx]['Volume']
            subsampled.g1s_size_measured = decimated_ts.iloc[g1s_idx]['Measured volume']
            subsampled.g1s_time = decimated_ts.iloc[g1s_idx]['Time']
            subsampled.g1s_frame = g1s_idx
        
        if self.divided:
            division_idx = decimated_ts['Time'].argmax()
            # Division
            subsampled.div_size = decimated_ts.iloc[division_idx]['Volume']
            subsampled.div_size_measured = decimated_ts.iloc[division_idx]['Measured volume']
            subsampled.div_time = decimated_ts.iloc[division_idx]['Time']
            subsampled.div_frame = division_idx

        # Recalculate duration + growth
        subsampled.g1_duration = subsampled.g1s_time - subsampled.birth_time
        subsampled.sg2_duration = subsampled.div_time - subsampled.birth_time
        subsampled.total_duration = subsampled.div_time - subsampled.g1s_time
        
        return subsampled


# --- DATA METHODS ----
    # def decatenate_fields()
    def __repr__(self):
        string = 'Cell ID = ' + str(self.cellID) + '\n'
        string += 'Born at frame ' + str(self.birth_frame) + '\n'
        string += 'Divided : ' + str(self.divided) + '\n\n\n'
        return string

