#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:47:50 2019

@author: xies
"""


# Print histogram of durations
bins = np.arange(11) - 0.5
plt.figure()
plt.hist((df['G1 length'])/12,bins,histtype='step')
plt.hist((df['Cycle length'] - df['G1 length'])/12,bins,histtype='step')
plt.xlabel('Phase duration (frames)')

plt.figure()
plt.hist((df['G1 length']),histtype='step')
plt.hist((df['Cycle length'] - df['G1 length']),histtype='step')
plt.xlabel('Phase duration (hr)')

########
plt.figure()
plt.hist(df['Cycle length'],9)
plt.vlines(df['Cycle length'].mean(),0,45)
plt.ylim([0,45])
plt.xlabel('Cell cycle duration (hr)')
