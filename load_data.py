############################################
# This is util for loading training datas.
###########################################

from config import *

class DataLoader(object):
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_data = dict()

    def _load_data(self):
	print "READ  ",self.file_path," ..."
	lines = -1
        for line in file(self.file_path,"r"):
	    lines += 1
	    if lines > 0 :
	        segs = line.strip().split(",")
		features = []
		for idx in range(len(segs)):
		    if idx in NUMERIC_COLS:
			features.append( float(segs[idx]) )
		self.file_data[lines] = features

    def _test_data(self):
        print "lines:",len( self.file_data.keys() )
        ids = [ 1, 2, 1000, 2000]
	for idx in ids:
	    print "features:",self.file_data[idx]
        
    def _next_batch( bat_size ):
        print bat_size

    

