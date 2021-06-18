#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:45:44 2021

@author: xies
"""

class Track:
    def __init__(self,trackID,root):
        self.trackID = trackID
        self.root = root
        self.preliminary = True
        
        
        
        
class Spot:
    def __init__(self,spotID,coordinate):
        self.id = spotID
        x,y,z,frame = coordinate
        self.x = x
        self.y = y
        self.z = z
        self.frame = frame
        self.linked = False
        
    def link(self,left,right):
        self.left = left
        self.right = right
        # Catergorize this linkage
        if (left == None) and (right == None):
            # No children -> end of track
            self.terminal = True
            self.division = False
        elif (left == None) or (right == None):
            # Only one child -> part of stem
            self.division = False
            self.terminal = False
        else:
            # Two children
            self.termianl = False
            self.division = True
        self.linked = True
        