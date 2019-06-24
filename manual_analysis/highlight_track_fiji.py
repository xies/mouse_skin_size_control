from ij import IJ, ImagePlus, ImageListener, CompositeImage
from ij.gui import PointRoi, NonBlockingGenericDialog
from net.imglib2.converter import Converters
from net.imglib2.realtransform import RealViews as RV  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from net.imglib2.view import Views  
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory  
import os, sys
from collections import defaultdict
from java.awt import Color

# Grab open image
imp = IJ.getImage() # Replace with actual IO dialog later
img = CompositeImage(imp)

# Create a listener that, on slice change, updates the ROI  
class PointRoiRefresher(ImageListener):
	def __init__(self, imp, nuclei):  
		self.imp = imp  
		# A map of slice indices and 2D points, over the whole 4d volume  
		self.nuclei = defaultdict(list)  # Any query returns at least an empty list
		for frame,coord in nuclei.iteritems():
			self.nuclei[frame] = coord
	def imageOpened(self, imp):
		pass
	def imageClosed(self, imp):
		if imp == self.imp:
			imp.removeImageListener(self)
	def imageUpdated(self, imp):  
		if imp == self.imp:
			self.updatePointRoi()
	def updatePointRoi(self):
    # Surround with try/except to prevent blocking  
    #   ImageJ's stack slice updater thread in case of error.  
		try:  
		# Update PointRoi  
			self.imp.killRoi()
			point = self.nuclei[self.imp.getFrame()] # map 1-based slices  
                                                   # to 0-based nuclei Z coords
            
			if len(point) == 0:
				return
			IJ.log("Cell found in frame " + str(self.imp.getFrame()))
			# New empty PointRoi for the current slice  
			roi = PointRoi(point[0],point[1])
			# Style: large, red dots  
			roi.setSize(4) # ranges 1-4  
			roi.setPointType(2) # 2 is a dot (filled circle)  
			roi.setFillColor(Color.red)  
			roi.setStrokeColor(Color.red)
			self.imp.setRoi(roi)
		except:
			IJ.error(sys.exc_info())  
  
# Read CSV files as list of int
def txt2list(filename):
	f = open(filename,'r')
	return [int(i) for i in f.readlines()]

# --- MAIN ----

# Load coordinates for each cell
basedir = '/Users/mimi/Box Sync/Mouse/Skin/W-R5/tracked_cells/'
celldirs = os.listdir(basedir)
celldirs = [d for d in celldirs if os.path.isdir(os.path.join(basedir,d))]

# Read in the log file
log_filename = os.path.join(basedir, 'log.txt')
if not os.path.exists(log_filename):
	with open(log_filename,'w') as f:
		f.write("cIDs already done\n")
		already_done = []
else:
	with open(log_filename,'r') as f:
		already_done = [line for line in f.readlines()][1:]
		already_done = [line.strip() for line in already_done]

nuclei = {}
for cID in celldirs:
	
	# Check logfile if this cID already is processed
	if cID in already_done:
		print 'Skipping ', cID
		continue

	thisdir = os.path.join(basedir, cID)
	os.chdir(thisdir)
	
	# Read csv files into lists, create nuclei list
	T = txt2list( os.path.join(thisdir,'t.csv') )
	X = txt2list( os.path.join(thisdir,'x.csv') )
	Y = txt2list( os.path.join(thisdir,'y.csv') )
	nuclei = defaultdict(list)
	for i,t in enumerate(T):
		nuclei[t] = ( X[i],Y[i] )
	print 'Working on: ', cID
	# Add listener to image
	listener = PointRoiRefresher(imp, nuclei)
	ImagePlus.addImageListener(listener)
	
	# Wait for user to clear current cell
	gd = NonBlockingGenericDialog("Advance to next cell?")
	gd.setOKLabel('Next')
	gd.setCancelLabel('Completely exit')
	gd.addCheckbox('Skip this cell?',False)
	gd.showDialog()
	if gd.wasOKed(): # OK means 'advance'
		imp.removeImageListener(listener)
		if gd.getNextBoolean():
			with open(os.path.join(thisdir,'skipped.txt'),'w') as f:
				f.write('Skipped')
		# Log the cell as done
		with open(log_filename, 'a') as f:
			f.write(''.join((cID,'\n')))
		print "Advancing to next cell..."
		continue
	
	if gd.wasCanceled(): # Cancel means 'exit'
		imp.removeImageListener(listener)
		print "Exiting completely."
		break

	

	