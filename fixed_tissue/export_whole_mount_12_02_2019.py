from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
from glob import glob
from os import path

dirnames = ['/Users/xies/Box/mIOs/Confocal/11-22-2019 RB-KO FUCCIC/EtOH/DAPI FUCCI-C Olfm4-594 betaCat-647',
'/Users/xies/Box/mIOs/Confocal/11-22-2019 RB-KO FUCCIC/4OHT 48hr/DAPI FUCCI-C Olfm4-594 betaCat-647']
channels = {'dapi':2}


# Iterate through all region directoryes
for directory in dirnames:
	for filename in glob(path.join(directory,'*.tif')):
		# Open image
		print filename
		im = IJ.openImage(filename)
		im.show()

		# Iterate through channels with > tool:
		for chan in channels.iterkeys():
	
			for i in range(channels[chan]):
				IJ.run("Next Slice [>]")
	
			roiMan = RoiManager.getInstance()
			
			# glob all subdir within /tracked_cells
			subdirlist = glob(path.join(directory,'*'))
			print subdirlist
			for subdir in subdirlist:
				# Grab all the .zip files and load into RoiManager
				ziplist = glob(path.join(subdir,'*.zip'))
				for zipname in ziplist:
					# Measure FUCCI & save to .txt
					roiMan.runCommand('Open',zipname)
					roiMan.runCommand('Select All')
					roiMan.runCommand('Measure')
					rt = ResultsTable.getResultsTable()
					fucci_savename = ''.join( (path.splitext(zipname)[0],'.',chan,'.txt') )
					rt.save(fucci_savename)
					roiMan.reset()
					rt.reset()
				
		im.close()

		