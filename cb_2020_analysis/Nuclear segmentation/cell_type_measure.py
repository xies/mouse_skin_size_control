from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
from glob import glob
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Mesa et al/W-R2'
channels = ['fucci','actin','h2b'] # Channel names

im = IJ.openImage(path.join(dirname,'20161127_Fucci_1F_0-168hr_R2.tif'))
im.show()

# Iterate through channels with > tool:
for chan in channels:

	roiMan = RoiManager.getInstance()
	
	# glob all subdir within /tracked_cells
	for ROIs in glob(path.join(dirname,'Stem v diff/*.zip')):
		# Measure FUCCI & save to .txt
		roiMan.runCommand('Open',ROIs)
		roiMan.runCommand('Select All')
		roiMan.runCommand('Measure')
		rt = ResultsTable.getResultsTable()
		channel_savename = ''.join( (path.splitext(ROIs)[0],'.',chan,'.txt') )
		rt.save(channel_savename)
		roiMan.reset()
		rt.reset()
	# Advance to next channel
	IJ.run("Next Slice [>]")		

im.close()

		