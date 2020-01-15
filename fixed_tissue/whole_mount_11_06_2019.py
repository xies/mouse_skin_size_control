from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
from glob import glob
from os import path

dirnames = ['/Users/xies/Box/Mouse/Skin/Fixed/11-06-2019 Skin Ecad488 EdU8h/WT/',
	'/Users/xies/Box/Mouse/Skin/Fixed/11-06-2019 Skin Ecad488 EdU8h/RB-KO/']
channels = {'dapi':'3.tif'}

# Iterate through all region directoryes
for regiondir in dirnames:
	#Iterate through all channels
	for chan_name in channels.iterkeys():
		# Open image
		im = IJ.openImage(path.join(regiondir,channels[chan_name]))
		roiMan = RoiManager.getInstance()
		im.show()

		# Grab all the .zip files and load into RoiManager
		filelist = glob(path.join(regiondir,'3/*.zip'))
		for fullname in filelist:
			print fullname
			# Measure FUCCI & save to .txt
			IJ.run("Next Slice [>]")
			roiMan.runCommand('Open',fullname)
			roiMan.runCommand('Select All')
			roiMan.runCommand('Measure')
			rt = ResultsTable.getResultsTable()
			fucci_savename = ''.join( (path.splitext(fullname)[0],'.', chan_name ,'.txt') )
			rt.save(fucci_savename)
			roiMan.reset()
			rt.reset()
			
		im.close()