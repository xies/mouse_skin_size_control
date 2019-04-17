from ij import IJ, ImagePlus, ImageListener, CompositeImage
from ij.gui import PointRoi
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
				IJ.log("No points for frame " + str(self.imp.getFrame()))  
				return
			# New empty PointRoi for the current slice  
			roi = PointRoi(point[0],point[1])
			# Style: large, red dots  
			roi.setSize(4) # ranges 1-4  
			roi.setPointType(2) # 2 is a dot (filled circle)  
			roi.setFillColor(Color.red)  
			roi.setStrokeColor(Color.red)
			self.imp.setRoi(roi)
		except IOError:
			print "HMMM"
#			IJ.error(sys.exc_info())  
  

def csv2list(filename):
	f = open(filename,'r')
	return [int(i) for i in f.readlines()]

# Load coordinates for each cell
basedir = '/Users/mimi/Box Sync/Mouse/Skin/W-R1/tracked_cells/'
celldirs = os.listdir(basedir)

roi = PointRoi(0,0)
roi.setSize(4)
roi.setPointType(2)
roi.setFillColor(Color.red)

nuclei = {}
for cID in celldirs[0:1]:
	print 'Working on: ', cID
	thisdir = os.path.join(basedir, cID)
    
	# Read csv files into lists
	T = csv2list( os.path.join(thisdir,'t.csv') )
	X = csv2list( os.path.join(thisdir,'x.csv') )
	Y = csv2list( os.path.join(thisdir,'y.csv') )

for i,t in enumerate(T):
	nuclei[t] = ( X[i],Y[i] )

point = nuclei[t]
print point
#roi = PointRoi(point[0],point[1])
# Style: large, red dots  
#roi.setSize(4) # ranges 1-4  
#roi.setPointType(2) # 2 is a dot (filled circle)  
#roi.setFillColor(Color.red)  
#roi.setStrokeColor(Color.red)
#imp.setRoi(roi)

listener = PointRoiRefresher(imp, nuclei)
ImagePlus.addImageListener(listener)

#	first_frame = min(0, int(T[0])-1)
#	last_frame = min(0, T[0]-1)
	