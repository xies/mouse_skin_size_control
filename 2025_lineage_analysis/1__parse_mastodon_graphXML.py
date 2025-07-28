#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:17:36 2025

@author: xies
"""

from os import path
import networkx as nx

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/Mastodon/'

#%% Read the branches

def extract_subdigraph(G,nodes):
    edges = nx.dfg_

G = nx.read_graphml(path.join(dirname,'all_branches.xml'))

# Subgraph individual lineages
num_lineages = nx.number_weakly_connected_components(G)
lineages = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

# Key each lineage using the 'parent' SpotID