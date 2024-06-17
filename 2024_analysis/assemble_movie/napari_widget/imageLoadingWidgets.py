#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:33:54 2024

@author: xies
"""

# Image loader class
import numpy as np
import napari
from magicgui.widgets import ComboBox, Container, FileEdit, PushButton, RadioButtons, LineEdit
from itertools import cycle
from typing import List

from napari.utils.notifications import show_warning

# from os import path
from skimage import io

from twophotonUtils import parse_unregistered_channels, parse_unaligned_channels, parse_aligned_timecourse_directory

from os import path
from glob import glob

class LoadTimepointForInspection(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._timepoint_choices = ()
        self._set_choices = ['Un-registered']
        self._viewer = viewer
        self.append(
            FileEdit(name="dirname",label="Image region to load:",mode="d")
        )
        self.append(
            LineEdit(name='pattern_str',label='subdir filter',value='*. Day*/')
        )
        self.append(
            ComboBox(name="timepoint2load", choices=self.get_timepoint_choices, label="Select timepoint")
        )
        self.append(RadioButtons(name='set2load',
                                  choices=self.get_set_choices,
                                  value='Un-registered'
                                  )
                    )
        self.append(PushButton(name='load_button', text='Load timepoint'))

        self.dirname.changed.connect(self.update_timepoint_choices)
        self.pattern_str.changed.connect(self.update_timepoint_choices)
        self.timepoint2load.changed.connect(self.update_set_choices)
        self.load_button.changed.connect(self.load_images)

    def get_timepoint_choices(self, dropdown_widget):
        return self._timepoint_choices

    def update_timepoint_choices(self):
        dirname = self.dirname.value
        pattern_str = self.pattern_str.value
        choices = None
        filelist = parse_unregistered_channels(dirname,folder_str=pattern_str)
        if len(filelist) > 0:
            choices = filelist.index.values
        else:
            show_warning(f'Directory {dirname} is not a region directory.')
        if choices is not None:
            self._timepoint_choices = choices
            self.timepoint2load.reset_choices()

    def get_set_choices(self, radio_widget):
        return self._set_choices

    def update_set_choices(self):
        dirname = self.dirname.value
        t = self.timepoint2load.value
        subdir_name = glob(path.join(dirname,f'{t}. */'))[0]

        choices = []
        if len( glob(path.join(subdir_name,'R_reg.tif'))) > 0:
            choices.append('Un-registered')
        if len( glob(path.join(subdir_name,'R_reg_reg.tif'))) > 0:
            choices.append('Registered')
        if len( glob(path.join(subdir_name,'R_align.tif'))) > 0:
            choices.append('Aligned')
        self._set_choices = choices
        self.set2load.reset_choices()


    def load_images(self) -> List[napari.layers.Layer]:
        # clear the current slate
        names2remove = [l.name for l in self._viewer.layers]
        for l in names2remove:
            self._viewer.layers.remove(l)

        prefix = f'{self.timepoint2load.value}_'

        if self.set2load.value == 'Un-registered':
            filelist = parse_unregistered_channels(self.dirname.value,folder_str=self.pattern_str.value) #@todo: auto-detect when these are not yet available
            suffix = '_reg'
        elif self.set2load.value == 'Registered':
            filelist = parse_unaligned_channels(self.dirname.value,folder_str=self.pattern_str.value)
            suffix = '_reg_reg'
        elif self.set2load.value == 'Aligned':
            filelist = parse_aligned_timecourse_directory(self.dirname.value,folder_str=self.pattern_str.value)
            suffix = '_align'

        # Load the files using a cycling colormap series
        file_tuple = filelist.loc[self.timepoint2load.value]
        colormaps = cycle(['bop blue','gray','bop orange','bop purple'])

        for name,filename in file_tuple.items():
            self._viewer.add_image(io.imread(filename),name=prefix+name+suffix, blending='additive', colormap=next(colormaps))

class LoadChannelForInspection(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._channel_choices = ()
        self._set_choices = ['Un-registered']
        self._viewer = viewer
        self.append(
            FileEdit(name="dirname",label="Image region to load:",mode="d")
        )
        self.append(
            LineEdit(name='pattern_str',label='subdir filter',value='*. Day*/')
        )
        self.append(
            ComboBox(name="channel2load", choices=self.get_channel_choices, label="Select channel")
        )
        self.append(RadioButtons(name='set2load',
                                  choices=self.get_set_choices,
                                  value='Un-registered'
                                  )
                    )
        self.append(PushButton(name='load_button', text='Load timepoint'))

        self.dirname.changed.connect(self.update_channel_choices)
        self.pattern_str.changed.connect(self.update_channel_choices)
        self.channel2load.changed.connect(self.update_set_choices)
        self.load_button.changed.connect(self.load_images)

    def get_channel_choices(self, dropdown_widget):
        return self._channel_choices

    def update_channel_choices(self):
        dirname = self.dirname.value
        pattern_str = self.pattern_str.value
        choices = None

        filelist = parse_unregistered_channels(dirname,folder_str=pattern_str)
        if len(filelist) > 0:
            choices = filelist.columns
        else:
            show_warning(f'Directory {dirname} is not a region directory.')
        if choices is not None:
            self._channel_choices = choices
            self.channel2load.reset_choices()

    def get_set_choices(self, radio_widget):
        return self._set_choices

    def update_set_choices(self):
        dirname = self.dirname.value
        t = self.channel2load.value

        choices = []
        if len( glob(path.join(dirname,f'*.*/R_reg.tif'))) > 0:
            choices.append('Un-registered')
        if len( glob(path.join(dirname,f'*.*/R_reg_reg.tif'))) > 0:
            choices.append('Registered')
        if len( glob(path.join(dirname,f'*.*/R_align.tif'))) > 0:
            choices.append('Aligned')
        self._set_choices = choices
        self.set2load.reset_choices()

    def load_images(self) -> List[napari.layers.Layer]:
        # clear the current slate
        names2remove = [l.name for l in self._viewer.layers]
        for l in names2remove:
            self._viewer.layers.remove(l)

        prefix = f'{self.channel2load.value}_'

        if self.set2load.value == 'Un-registered':
            filelist = parse_unregistered_channels(self.dirname.value,folder_str=self.pattern_str.value) #@todo: auto-detect when these are not yet available
            suffix = '_reg'
        elif self.set2load.value == 'Registered':
            filelist = parse_unaligned_channels(self.dirname.value,folder_str=self.pattern_str.value)
            suffix = '_reg_reg'
        elif self.set2load.value == 'Aligned':
            filelist = parse_aligned_timecourse_directory(self.dirname.value,folder_str=self.pattern_str.value)
            suffix = '_align'

        # Load the files using a cycling colormap series
        file_tuple = filelist[self.channel2load.value]
        colormaps = cycle(['bop blue','gray','bop orange','bop purple'])

        for t,filename in file_tuple.items():
            self._viewer.add_image(io.imread(filename),name=f'{t}_{self.channel2load.value}', blending='additive', colormap=next(colormaps))
