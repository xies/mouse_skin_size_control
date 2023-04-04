#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 20:27:08 2022

@author: xies
"""

from basicUtils import *
from numpy import random

Niter = 100
sigma = 0.1

# Subsampling -> growth rate estimation
sgr_theo = {}
sgr_est = {}
sgr_est_noise = {}
max_age = [1,2,3,4,5,6,7]
colors = [colormaps['viridis'](i) for i in np.linspace(1,255,8)]
for j,age in enumerate(max_age):
    
    # Assume 70h cell cycle -> every 12h -> ~6 average sampling rate
    num_frame = int(age * 2)
    t = np.arange(0,age * 24,12)
    Y = np.zeros((Niter,len(t)))
    y = np.exp( np.log(2) * t / (age*24))
    sgr = (np.diff(y) / y[:-1])/12
    sgr_est[age * 24] = sgr[0]
    print(colors[j])
                 
    for i in range(Niter):
        y = np.exp( np.log(2) * t / (age*24))
        y = y + sigma * random.randn(len(y))
        Y[i,:] = y
        
        y = Y.mean(axis=0)
        sgr = (np.diff(y) / y[:-1]) / 12
        sgr_theo[age*24] = np.log(2) / (age*24)
        sgr_est_noise[age * 24] = sgr[0]
        
    
        plt.scatter(y[:-1],sgr,color=colors[j],alpha=0.1)
        
    
#%%plt.figure()
plt.plot(sgr_est.keys(),sgr_theo.values(),'k*')
plt.plot(sgr_est.keys(),sgr_est.values(),'k*-')
plt.plot(sgr_est_noise.keys(),sgr_est_noise.values(),'k--')

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)

cells = pd.DataFrame()
cells['Age'] = df_.groupby('basalID').max()['Age']
cells['Min size'] = df_.groupby('basalID').min()['Volume (sm)']
cells['sgr'] = df_.groupby('basalID').mean()['Specific GR b (sm)']
cells['basalID'] = df_.groupby('basalID').mean().index
plt.scatter(jitter(cells['Age'],.1),cells['sgr'], alpha=0.5)
plot_bin_means(cells['Age'],cells['sgr'],bin_edges = np.linspace(20,150,12),
               minimum_n = 5, color='r')

plt.xlabel('Total cell cycle duration (hr)')
plt.ylabel('Mean SGR')
plt.legend(['Theory','Bias due to sampling',
    'Sampling bias (perfect exp growth + fixed noise)',
            'Individual cells','Binned mean'])

#%%

from scipy.stats import binned_statistic

bin_means,bin_edges,which_bin = binned_statistic(cells['Min size'],cells['Min size'])
cells['Min size bin'] = which_bin

for i,row in df_.iterrows():
    df_.at[i,'Max age'] = cells.loc[row['basalID']]['Age']
    df_.at[i,'Min size bin'] = cells.loc[row['basalID']]['Min size bin']


sb.lmplot(data=df_,x='Volume (sm)',y = 'Specific GR b (sm)',hue='Min size bin')
sb.lmplot(data=df_,x='Volume (sm)',y = 'Specific GR b (sm)',hue='Max age')

#%%
from scipy.optimize import curve_fit

Niter = 100
t = np.linspace(0,70,12)
sigma = 0.1

df = pd.DataFrame()
df.index = range(Niter)

for i,g in enumerate(range(Niter)):
    growth_rate = 0.01 * (1 + 0.5*random.randn())
    y = np.exp(t * growth_rate) + sigma * random.randn(len(t))
    
    
    plt.plot(t,y)
    df.at[i,'Growth rate'] = growth_rate
    
    df.at[i,'Fitted k'] = curve_fit(lambda x,k: np.exp(k*t),t,y,0.1)[0]
    df.at[i,'SGR'] = (np.diff(y) / y[:-1]).mean() /12 # hr per frame
    





