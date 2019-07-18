from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
import os
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/W-R5/'

# Open FUCCI image
fucci = IJ.openImage(path.join(dirname,'fucci_cropped.tif'))
roiMan = RoiManager.getInstance()
fucci.show()

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

            # Measure FUCCI & save to .txt
            if path.splitext(fullname)[1] == '.zip':
                roiMan.runCommand('Open',fullname)
                roiMan.runCommand('Select All')
                roiMan.runCommand('Measure')
                rt = ResultsTable.getResultsTable()
                fucci_savename = ''.join( (path.splitext(fullname)[0],'.fucci.txt') )
                rt.save(fucci_savename)
                roiMan.reset()
                rt.reset()
				