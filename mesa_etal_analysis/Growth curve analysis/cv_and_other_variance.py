#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:49:27 2019

@author: xies
"""

df = df_has_daughter
Ncells = len(df)

# Plot CV/variation
# Fraction of growth in G1
plt.figure()
plt.hist(df['G1 grown']/df['Total growth'])
plt.xlabel('Fraction of growth occuring in G1')
plt.ylabel('Frequency')

# Calculate CV CIs parametrically
[birthCV,bCV_lcl,bCV_ucl] = cvariation_ci(df['Birth volume'])
bCV_lci = birthCV - bCV_lcl; bCV_uci = bCV_ucl - birthCV
[g1CV,gCV_lcl,gCV_ucl] = cvariation_ci(df['G1 volume'])
gCV_lci = g1CV - gCV_lcl; gCV_uci = gCV_ucl - g1CV
[divisionCV,dCV_lcl,dCV_ucl] = cvariation_ci(df['Division volume interpolated'])
dCV_lci = divisionCV - dCV_lcl; dCV_uci = dCV_ucl - divisionCV

# Bootstrap CVs -- bootstrap at cell level and back out diff in mean
Nboot = 10000
bCV_ = np.zeros(Nboot)
gCV_ = np.zeros(Nboot)
dCV_ = np.zeros(Nboot)
bdCV_diff = np.zeros(Nboot)
bgCV_diff = np.zeros(Nboot)
gdCV_diff = np.zeros(Nboot)
for i in xrange(Nboot):
    # Random resample w/ replacement at cell level allows for CVs to be compared
    df_ = pd.DataFrame(df.values[random.randint(Ncells, size=Ncells)], columns=df.columns)
    bCV_[i] = stats.variation(df_['Birth volume'])
    gCV_[i] = stats.variation(df_['G1 volume'])
    dCV_[i] = stats.variation(df_['Division volume interpolated'])
    bdCV_diff[i] = bCV_[i] - dCV_[i]
    bgCV_diff[i] = bCV_[i] - gCV_[i]
    gdCV_diff[i] = gCV_[i] - dCV_[i]

bCV_lcl,bCV_ucl = stats.mstats.mquantiles(bCV_,prob=[0.05,0.95])
gCV_lcl,gCV_ucl = stats.mstats.mquantiles(gCV_,prob=[0.05,0.95])
dCV_lcl,dCV_ucl = stats.mstats.mquantiles(dCV_,prob=[0.05,0.95])

plt.hist(bdCV_diff)
plt.vlines(birthCV-divisionCV,0,2500)
plt.figure()
plt.hist(bgCV_diff)
plt.vlines(birthCV-g1CV,0,2500)

plt.figure()
plt.hist(gdCV_diff)
plt.vlines(g1CV-divisionCV,0,2500)

errors = np.array(((bCV_lci,bCV_uci),(gCV_lci,gCV_uci),(dCV_lci,dCV_uci))).T
plt.figure()
plt.errorbar([1,2,3],[birthCV,g1CV,divisionCV],
             yerr=errors,fmt='o',ecolor='orangered',
            color='steelblue', capsize=5)
plt.xticks([1,2,3,4],['Birth volume','G1 volume','Division volume'])
plt.ylabel('Coefficient of variation')


# Calculate if bCV > gCV in bootstrap

print 'P(birth CV > div CV) = ', float((bdCV_diff > 0).sum()) / len(bdCV_diff)
print 'P(birth CV > g1 CV) = ', float((bgCV_diff > 0).sum()) / len(bgCV_diff)
print 'P(g1 CV > div CV) = ', float((gdCV_diff > 0).sum()) / len(gdCV_diff)

# Calculate dispersion index
#birthFano = np.var(df['Birth volume']) / np.mean(df['Birth volume'])
#g1Fano = np.var(df['G1 volume']) / np.mean(df['G1 volume'])
#divisionFano = np.var(df['Division volume']) / np.mean(df['Division volume'])

# Calculate skew
birthSkew = stats.skew(df['Birth volume'])
g1Skew = stats.skew(df['G1 volume'])
divisionSkew = stats.skew(df['Division volume'])

sb.catplot(data=df.melt(id_vars='CellID',value_vars=['Birth volume','G1 volume','Division volume'],
                        value_name='Volume'),
           x='variable',y='Volume')

plt.figure()
plt.plot([birthCV,g1CV,divisionCV])
plt.xticks(np.arange(3), ['Birth volume','G1 volume','Division volume'])
plt.ylabel('Coefficient of variation')

plt.figure()
plt.plot([birthSkew,g1Skew,divisionSkew])
plt.xticks(np.arange(3), ['Birth volume','G1 volume','Division volume'])
plt.ylabel('Distribution skew')
