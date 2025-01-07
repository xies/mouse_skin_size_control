#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:45:33 2024

@author: xies
"""

from glob import glob

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-23-2023 R26CreER Rb-fl no tam ablation'

ablation_sphase = []
ablation_total = []
nonablation_sphase = []
nonablation_total = []

filelist = glob(path.join(dirname,'*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[1])
filelist = glob(path.join(dirname,'*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[1])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[1])
filelist = glob(path.join(dirname,'*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[1])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)
print(f'M1: ablation fraction: {len(ablation_sphase)/len(ablation_total)}; total = {len(ablation_total)}')
print(f'M1: non ablation fraction: {len(nonablation_sphase)/len(nonablation_total)}; total = {len(nonablation_total)}')

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/'

ablation_sphase = []
ablation_total = []
nonablation_sphase = []
nonablation_total = []

filelist = glob(path.join(dirname,'*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)

print(f'M2: ablation fraction: {len(ablation_sphase)/len(ablation_total)}; total = {len(ablation_total)}')
print(f'M2: non ablation fraction: {len(nonablation_sphase)/len(nonablation_total)}; total = {len(nonablation_total)}')

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black unclipped less leaky DOB 06-30-2023/'
ablation_sphase = []
ablation_total = []
nonablation_sphase = []
nonablation_total = []

filelist = glob(path.join(dirname,'*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)

print(f'M3: ablation fraction: {len(ablation_sphase)/len(ablation_total)}; total = {len(ablation_total)}')
print(f'M3: non ablation fraction: {len(nonablation_sphase)/len(nonablation_total)}; total = {len(nonablation_total)}')

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/'
ablation_sphase = []
ablation_total = []
nonablation_sphase = []
nonablation_total = []

filelist = glob(path.join(dirname,'*/*/*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/*/*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/*/*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[1])
filelist = glob(path.join(dirname,'*/*/*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[1])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/*/*/Ablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[2])
filelist = glob(path.join(dirname,'*/*/*/Ablation_S_phase.csv'))
sphase = pd.read_csv(filelist[2])
ablation_sphase.extend(sphase.index)
ablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/*/*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[0])
filelist = glob(path.join(dirname,'*/*/*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[0])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)

filelist = glob(path.join(dirname,'*/*/*/Nonablation_all_cells.csv'))
all_cells = pd.read_csv(filelist[1])
filelist = glob(path.join(dirname,'*/*/*/Nonablation_S_phase.csv'))
sphase = pd.read_csv(filelist[1])
nonablation_sphase.extend(sphase.index)
nonablation_total.extend(all_cells.index)

print(f'M4: ablation fraction: {len(ablation_sphase)/len(ablation_total)}; total = {len(ablation_total)}')
print(f'M4: non ablation fraction: {len(nonablation_sphase)/len(nonablation_total)}; total = {len(nonablation_total)}')


#%%
df = pd.DataFrame()

df.at['M1','Ablation'] = 0.323943661971831
df.at['M1','Nonablation'] = 0.22413793103448276
df.at['M2','Ablation'] = 0.5526315789473685
df.at['M2','Nonablation'] = 0.5238095238095238
df.at['M3','Ablation'] = 0.5333333333333333
df.at['M3','Nonablation'] = 0.22727272727272727
df.at['M4','Ablation'] =  0.19811320754716982
df.at['M4','Nonablation'] = 0.15384615384615385

df = df.reset_index(names='Mouse')
df = df.melt(id_vars='Mouse')

sb.barplot(df,x='Mouse',y='value',hue='variable')

plt.ylabel('Fraction of cells entering S phase')
