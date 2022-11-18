#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:02:43 2022

@author: xies
"""

import numpy as np
from skimage import io
from os import path, stat
from glob import glob
import pandas as pd
import seaborn as sb
from re import match
import matplotlib.pylab as plt

# Custom packages
from basicUtils import nonans, nonan_pairs, plot_bin_means, jitter

df_ = wt; title_str = 'WT'; subplot = 1
# df_ = ko; title_str = 'RB-KO'; subplot= 2

#% Some quality control plots. Some time frames are not as good as others

# sb.lmplot(data = df_, x = 'Birth frame', y = 'Birth size', hue='Region')
# sb.lmplot(data = df_, x = 'S phase frame',y = 'S phase size', hue='Region')
# sb.lmplot(data = df_, x='Division frame',y='Division size',hue='Region')

#%% Size homeotasis plots

plt.figure(1)
plt.subplot(1,3,subplot)
# plt.scatter(df_['Birth size'],df_['G1 growth'])
sb.regplot(data = df_, x = 'Birth size', y = 'G1 growth', robust=False)
plot_bin_means(df_['Birth size'],df_['G1 growth'],bin_edges=9,bin_style='equal',minimum_n=5)
plt.gca().axis('equal')
plt.xlabel('Birth size (fL)'); plt.ylabel('G1 growth (fL)')
plt.title(title_str)

plt.figure(2)
plt.subplot(1,3,subplot)
# plt.scatter(df_['Birth size'],df_['Total growth'])
sb.regplot(data = df_, x = 'Birth size', y = 'Total growth', robust=False)
plot_bin_means(df_['Birth size'],df_['Total growth'],bin_edges=9,bin_style='equal',minimum_n=5)
plt.gca().axis('equal')

plt.xlabel('Birth size (fL)'); plt.ylabel('Total growth (fL)')
plt.title(title_str)

#% Durations
plt.figure(3)
plt.subplot(1,3,subplot)
# plt.scatter(df_['Birth size'],df_['G1 length'])
sb.regplot(data = df_,x='Birth size',y ='G1 length',y_jitter=2)
plot_bin_means(df_['Birth size'],df_['G1 length'],minimum_n=5,bin_edges=10,bin_style='equal')
plt.title(title_str)
plt.ylim([0,180])

plt.figure(4)
# plt.xlim([100,350]); plt.ylim([0,200])
plt.subplot(1,3,subplot)
# plt.scatter(df_['Birth size'],df_['Cycle length'])
sb.regplot(data = df_,x='Birth size',y ='Cycle length',y_jitter=2)
plot_bin_means(df_['Birth size'],df_['Cycle length'],minimum_n=5,bin_edges=10,bin_style='equal')
plt.legend(['G1 length','Total length'])
plt.xlabel('Birth size (fL)'); plt.ylabel('Cell cycle duration (h)')
plt.title(title_str)
plt.ylim([0,220])

#%% Compare w/ Mesa dataset

mesa_ = mesa; title_str = 'Mesa (N=197)'
# mesa_ = mesa[mesa['Region'] == 'M1R1']; title_str = 'Mesa region 1 (N=87)'

plt.figure(1)
plt.subplot(1,3,3)
sb.regplot(data = mesa_, x='Birth nuc volume',y='G1 nuc grown')
plot_bin_means(mesa_['Birth nuc volume'],mesa_['G1 nuc grown'],bin_edges=10,minimum_n=8,bin_style='equal')
plt.title(title_str)
# plt.gca().axis('equal')
plt.figure(2)
plt.subplot(1,3,3)
sb.regplot(data = mesa_, x='Birth nuc volume',y='Total nuc growth')
plot_bin_means(mesa_['Birth nuc volume'],mesa_['Total nuc growth'],bin_edges=10,minimum_n=8,bin_style='equal')
# plt.gca().axis('equal')
plt.title(title_str)

plt.figure(3)
plt.subplot(1,3,3)
sb.regplot(data = mesa_, x='Birth nuc volume',y='G1 length',y_jitter=2)
plot_bin_means(mesa_['Birth nuc volume'],mesa_['G1 length'],bin_edges=10,minimum_n=8,bin_style='equal')
plt.ylim([0,180])
plt.title(title_str)

plt.figure(4)
plt.subplot(1,3,3)
sb.regplot(data = mesa_, x='Birth nuc volume',y='Cycle length',y_jitter=2)
plot_bin_means(mesa_['Birth nuc volume'],mesa_['Cycle length'],bin_edges=10,minimum_n=8,bin_style='equal')
plt.ylim([0,220])
plt.title(title_str)

#%% Growth ratios

def bootstrap_field(df,field):
    N = len(df)
    I = np.random.choice(N, N)
    
    return df.iloc[I][field]

Nboot = 1000
ratio_bs = np.zeros((Nboot,len(df_)))
for i in range(Nboot):
    div_size_boot = bootstrap_field(df_,'Division size (sm)')
    ratio_bs[i,:] = div_size_boot / df_['Birth size'].values

plt.hist(ratio_bs.flat,density=True)
plt.hist(df_['Division size (sm)'] / df_['Birth size'],density=True)

#%% Sisters / daughters

def find_sister_pairs(df):
    assert(len(np.unique(df['Genotype'])) == 1)
    mothers = nonans(np.unique(df['MotherID']))
    sisters = [df[df['MotherID'] == m] for m in mothers]
    # filter out singletons
    sisters = [s for s in sisters if len(s) == 2]
               
    return sisters

sisters_wt = find_sister_pairs(wt)
sister_symmetry_wt = np.array([np.diff( sis['Birth size'] )[0] for sis in sisters_wt])
mean_birth_size_wt = np.array([np.mean( sis['Birth size']) for sis in sisters_wt])
sister_cyc_wt = np.array([np.diff(sis['Cycle length'])[0] for sis in sisters_wt])

sisters_ko = find_sister_pairs(ko)
sister_symmetry_ko = np.array([np.diff( sis['Birth size'] )[0] for sis in sisters_ko])
mean_birth_size_ko = np.array([np.mean( sis['Birth size']) for sis in sisters_ko])
sister_cyc_ko = np.array([np.diff(sis['Cycle length'])[0] for sis in sisters_ko])

plt.hist(np.abs(sister_symmetry_wt),histtype='step')
plt.hist(np.abs(sister_symmetry_ko),histtype='step')
plt.xlabel('Sister asymmetry (fl)')
plt.legend(['WT','RB-KO'])

plt.figure()
plt.hist(np.abs(sister_symmetry_wt)/mean_birth_size_wt,histtype='step')
plt.hist(np.abs(sister_symmetry_ko)/mean_birth_size_ko,histtype='step')
plt.xlabel('Sister asymmetry (% of mean birth size)')
plt.legend(['WT','RB-KO'])

plt.figure()
plt.hist(np.abs(sister_cyc_wt),histtype='step')
plt.hist(np.abs(sister_cyc_ko),histtype='step')
plt.xlabel('Difference in cycle length (h)')
plt.legend(['WT','RB-KO'])

plt.figure()
plt.scatter(sister_symmetry_wt,sister_cyc_wt)
plt.scatter(sister_symmetry_ko,sister_cyc_ko)

#%% Histogram of cell cycle times

plt.figure()
plt.hist(wt['G1 length'],histtype='step'); plt.hist(ko['G1 length'],histtype='step')
plt.xlabel('G1 length (h)'); plt.legend(['WT','RB-KO'])

plt.figure()
plt.hist(wt['Cycle length'],histtype='step'); plt.hist(ko['Cycle length'],histtype='step',bins=14)
plt.xlabel('Cycle length (h)'); plt.legend(['WT','RB-KO'])

#%% Print stats

print('---WT: Growth, correlation')

X,Y = nonan_pairs(wt['Birth size'],wt['G1 growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 growth, R = {R[1]}')
X,Y = nonan_pairs(wt['Birth size'],wt['Total growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = total growth, R = {R[1]}')


print('---Mesa: Growth, correlation')

X,Y = nonan_pairs(mesa['Birth nuc volume'],mesa['G1 nuc grown'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 growth, R = {R[1]}')
X,Y = nonan_pairs(mesa['Birth nuc volume'],mesa['Total nuc growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = total growth, R = {R[1]}')


print('---RBKO: Growth, correlation')

X,Y = nonan_pairs(ko['Birth size'],ko['G1 growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 growth, R = {R[1]}')
X,Y = nonan_pairs(ko['Birth size'],ko['Total growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = total growth, R = {R[1]}')


print('\n\nWT: --- Time, correlation')
      
X,Y = nonan_pairs(wt['Birth size'],wt['G1 length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 length, R = {R[1]}')
X,Y = nonan_pairs(wt['Birth size'],wt['Cycle length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = cycle length, R = {R[1]}')


print('RBKO: --- Time, correlation')
X,Y = nonan_pairs(ko['Birth size'],ko['G1 length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 length, R = {R[1]}')
X,Y = nonan_pairs(ko['Birth size'],ko['Cycle length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = cycle length, R = {R[1]}')


print('\n\nWT: --- Growth, regression')
X,Y = nonan_pairs(wt['Birth size'],wt['G1 growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = g1 growth, m = {p[0]}')
X,Y = nonan_pairs(wt['Birth size'],wt['Total growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = Total growth, m = {p[0]}')

print('RBKO: --- Growth, regression')
X,Y = nonan_pairs(ko['Birth size'],ko['G1 growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = g1 growth, m = {p[0]}')
X,Y = nonan_pairs(ko['Birth size'],ko['Total growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = Total growth, m = {p[0]}')


print('Mesa: --- Growth, regression')
X,Y = nonan_pairs(mesa['Birth nuc volume'],mesa['G1 nuc grown'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = g1 growth, m = {p[0]}')
X,Y = nonan_pairs(mesa['Birth nuc volume'],mesa['Total nuc growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = Total growth, m = {p[0]}')




