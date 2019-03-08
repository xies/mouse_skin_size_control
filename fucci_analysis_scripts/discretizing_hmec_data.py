#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:07:42 2019

@author: xies
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb

filename = '/data/HMEC DFB tracked data/PBS.csv'
pbs = pd.read_csv(filename,names = ['Tg1','Bsize'])

filename = '/data/HMEC DFB tracked data/Palbo.csv'
palbo = pd.read_csv(filename,names = ['Tg1','Bsize'])

# Plot the data
plt.subplot(1,2,1)
sb.regplot( pbs['Bsize'],pbs['Tg1'] )
plt.ylim([0,35])
plt.subplot(1,2,2)
sb.regplot( palbo['Bsize'],palbo['Tg1'] )
plt.ylim([0,35])

R_pbs = stats.pearsonr( pbs['Bsize'],pbs['Tg1'] )
R_palbo = stats.pearsonr( palbo['Bsize'],palbo['Tg1'] )

# Discretize Tg1 into %-iles
Tbins = np.linspace(0.,27., 10)
which_bin = np.digitize(pbs['Tg1'],Tbins)
pbs['Tg1 discrete'] = Tbins[which_bin-1]
which_bin = np.digitize(palbo['Tg1'],Tbins)
palbo['Tg1 discrete'] = Tbins[which_bin-1]


# Re-plot the data
plt.subplot(1,2,1)
sb.regplot( pbs['Bsize'],pbs['Tg1 discrete'] )
plt.subplot(1,2,2)
sb.regplot( palbo['Bsize'],palbo['Tg1 discrete'] )

R_pbs = stats.pearsonr( pbs['Bsize'],pbs['Tg1 discrete'] )
R_palbo = stats.pearsonr( palbo['Bsize'],palbo['Tg1 discrete'] )
