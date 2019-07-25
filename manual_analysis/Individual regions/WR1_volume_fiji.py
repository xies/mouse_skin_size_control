from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
import os
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/W-R1/'

# Open single-channel image
image = IJ.openImage(path.join(dirname,'h2b_mask_clean.tif'))
roiMan = RoiManager.getInstance()
image.show()

# Use os.walk to iterate through individual cells
for cellID, dirs, files in os.walk( path.join(dirname,'tracked_cells') ):
    this_celldir = path.join(dirname,cellID)
    # Check if skipped.txt exists
    if path.exists( path.join(this_celldir,'skipped.txt') ):
        continue
    else:
        for f in files:
            # Grab all the .zip files and load into RoiManager
            fullname = path.join(this_celldir,f)
			
            # Measure Image & save to .txt
            if path.splitext(fullname)[1] == '.zip':
                roiMan.runCommand('Open',fullname)
                roiMan.runCommand('Select All')
                roiMan.runCommand('Measure')
                rt = ResultsTable.getResultsTable()
                image_savename = ''.join( (path.splitext(fullname)[0],'.h2b.txt') )
                rt.save(image_savename)
                roiMan.reset()
                rt.reset()