from ij import IJ, ImagePlus
from ij.io import RoiDecoder
from ij.measure import ResultsTable
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from util.opencsv import CSVWriter
from ij.process import FloatProcessor
from os import path
from glob import glob
from array import zeros
import csv

region_names = ['/Users/xies/Box/Mouse/Skin/W-R1/',
				'/Users/xies/Box/Mouse/Skin/W-R2/']

for dirname in region_names:
	filelist = glob(path.join(dirname,'tracked_cells/*/t*[!a!b].zip'))
	roiMan = RoiManager.getInstance()
	
	px = []
	py = []
	frames = []
	zpos = []
	cellIDs = []
	
	# Load all track ROIs
	for f in filelist:
		roiMan.runCommand('Open',f)
		# Iterate through each PolygonRoi
		Nrois = roiMan.getCount()
		for i in range(Nrois):
			roi = roiMan.getRoi(i)
			poly = roi.getPolygon()
			px.append(poly.xpoints)
			py.append(poly.ypoints)
			
			cellIDs.append( path.split( path.split(f)[0] )[1] )
			frames.append(roi.getTPosition())
			zpos.append(roi.getZPosition())
			
		roiMan.reset()
	
	print filelist[0]
	# Export each measurement and export to CSV
	with open(path.join(dirname,'ROIs/polygon_x.csv'),'wb') as csvfile:
		writer = csv.writer(csvfile,delimiter=',')
		for x in px:
			writer.writerow(x)
	
	with open(path.join(dirname,'ROIs/polygon_y.csv'),'wb') as csvfile:
		writer = csv.writer(csvfile,delimiter=',')
		for y in py:
			writer.writerow(y)
	
	with open(path.join(dirname,'ROIs/frame.csv'),'wb') as f:
		for t in frames:
			f.write("%d\n" % t)
	
	with open(path.join(dirname,'ROIs/zpos.csv'),'wb') as f:
		for z in zpos:
			f.write("%d\n" % z)
	
	with open(path.join(dirname,'ROIs/cellIDs.csv'),'wb') as f:
		for c in cellIDs:
			f.write("%s\n" % c)
		
			
