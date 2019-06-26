import os,sys
from mpicbg.ij.plugin import NormalizeLocalContrast
from ij import IJ,ImagePlus
from ij.io import FileSaver
from ij.process import ImageStatistics as IS  
  

sourceDir = "/data/Skin/W-R5/seq"
targetDir = "/data/Skin/W-R5/normalized"
  

# A function that takes an input image and returns a contrast-normalized one  
def normalizeContrast(imp):  
  # The width and height of the box centered at every pixel:  
  blockRadiusX = 100 # in pixels  
  blockRadiusY = 100  
  # The number of standard deviations to expand to  
  stds = 2  
  # Whether to expand from the median value of the box or the pixel's value  
  center = True  
  # Whether to stretch the expanded values to the pixel depth of the image  
  # e.g. between 0 and 255 for 8-bit images, or e.g. between 0 and 65536, etc.  
  stretch = True  
  # Duplicate the ImageProcessor  
  copy_ip = imp.getProcessor().duplicate()  
  # Apply contrast normalization to the copy  
  NormalizeLocalContrast().run(copy_ip, 200, 200, stds, center, stretch)  
  # Return as new image  
  
  # Threshold image by mean
  options = IS.MEAN
  stats = IS.getStatistics(copy_ip, options, imp.getCalibration())

  copy_ip.subtract( int(stats.mean)/2 )
  return ImagePlus(imp.getTitle(), copy_ip)

# A function that takes a file path, attempts to load it as an image,  
# normalizes it, and saves it in a different directory  
def loadProcessAndSave(sourcepath, fn):
  try:  
    imp = IJ.openImage(sourcepath)  
    norm_imp = fn(imp) # invoke function 'fn', in this case 'normalizeContrast'  
    targetpath = os.path.join(targetDir, os.path.basename(sourcepath))  
    if not targetpath.endswith(".tif"):  
      targetpath += ".tif"  
    FileSaver(norm_imp).saveAsTiff(targetpath)  
  except:  
    print "Could not load or process file:", sourcepath  
    print sys.exc_info()  

# Strategy #2: let os.walk do all the work
print sourceDir
for root, directories, filenames in os.walk(sourceDir):
	for filename in filenames:
		loadProcessAndSave(os.path.join(root, filename), normalizeContrast)  
