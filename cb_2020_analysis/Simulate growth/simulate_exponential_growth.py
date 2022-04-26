#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:15:37 2019

@author: xies
"""

import math
import numpy as np
import pandas as pd
from numpy import random
import seaborn as sb
from scipy import stats,special

N = 10000
dt = 12 #hr

# 1) Pick birth volume
# Estimate birth volume distribution
mu,sigma = stats.norm.fit(df['Birth volume'])
V0 = stats.norm.rvs(size = N, loc = mu, scale = sigma)


# 2) Simulate growth using exponential growth (single parameter, no noise)

# Modulate G1/S transition with size

# G1/S params from logit regression
g1s_x0 =  -13.562453
g1s_beta = 0.024973

growth_slope = 0.0038989308647642737

# Estimate S/G2/M duration probability
x = nonans(df['SG2 length'])
weights = np.ones(len(x)) / len(x)
P_sg2_duration,sg2_duration_bins = np.histogram(x,bins = 5,weights=weights,density=True)

# Monte Carlo growth simulation
growth_curves = []
for i in range(N):
    divided = 0
    g1s = 0
    Vt = V0[i]
    t = 0
    
    while ~divided:
        t = t + dt
        # Increment volume by growth_slope
        Vt = Vt * math.exp(growth_slope * t)
        p_g1s = random.rand()
        
        if p_g1s < logit(  )
        
        
