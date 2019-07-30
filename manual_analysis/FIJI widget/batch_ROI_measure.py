from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
from glob import glob
from os import path

dirnames = ['/Users/xies/Box/Mouse/Skin/W-R1/',
			'/Users/xies/Box/Mouse/Skin/W-R2/',
			'/Users/xies/Box/Mouse/Skin/W-R5/',
			'/Users/xies/Box/Mouse/Skin/W-R5-full/']
channels = {'h2b':'h2b_mask.tif',
			'fucci':'fucci_sequence.tif'}

# Iterate through all region directoryes
for regiondir in dirnames:
	#Iterate through all channels
	for chan_name in channels.iterkeys():
		# Open image
		im = IJ.openImage(path.join(regiondir,channels[chan_name]))
		roiMan = RoiManager.getInstance()
		im.show()
		
		# glob all subdir within /tracked_cells
		subdirlist = glob(path.join(regiondir,'tracked_cells/*/'))
		for subdir in subdirlist:
			# Check if skipped.txt exists
			if path.exists( path.join(subdir,'skipped.txt') ):
				continue
			else:
				# Grab all the .zip files and load into RoiManager
				filelist = glob(path.join(subdir,'*.zip'))
				for fullname in filelist:
					print fullname
					# Measure FUCCI & save to .txt
					roiMan.runCommand('Open',fullname)
					roiMan.runCommand('Select All')
					roiMan.runCommand('Measure')
					rt = ResultsTable.getResultsTable()
					fucci_savename = ''.join( (path.splitext(fullname)[0],'.', chan_name ,'.txt') )
					rt.save(fucci_savename)
					roiMan.reset()
					rt.reset()
					
		im.close()