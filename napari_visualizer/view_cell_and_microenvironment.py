from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, ComboBox, FileEdit, Table
import napari
from napari.layers import Labels
import pandas as pd
from pathlib import Path


# Load 

viewer = napari.Viewer()

viewer.window.add_dock_widget(create_widget,area='left')
# viewer.window.add_dock_widget(auto_register_b_and_rshg(),area='left')

# viewer.window.add_dock_widget(LoadTimepointForInspection(viewer),area='right')
# viewer.window.add_dock_widget(transform_image, area='right')
# viewer.window.add_dock_widget(filter_gaussian3d, area='right')

# if __name__ == '__main__':
napari.run()
