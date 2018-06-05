############################################
# This is util for loading training datas.
###########################################

from config import *
import numpy as np

class DataLoader(object):
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_data = dict()
	self.total = 0
	self.count = 0

    def _load_data(self):
	print "READ  ",self.file_path," ..."
	lines = -2
        for line in file(self.file_path,"r"):
	    lines += 1
	    if lines >= 0 :
	        segs = line.strip().split(",")
		features = []
		for idx in range(len(segs)):
		    if idx in NUMERIC_COLS:
			features.append( float(segs[idx]) )
		self.file_data[lines] = features
	self.total = lines + 1

    
    def _next_batch( self, batch_size, feat_size ):
        X_matrix = np.empty( (batch_size,feat_size), dtype=np.float32 )
	self.count += 1
	stPos = ( batch_size * (self.count-1) ) % self.total
	ed = ( batch_size * self.count -1 ) % self.total
	edPos = -1
	if ed > stPos:
	    edPos = ed
	else:
	    edPos = self.total - 1
	for idx in range(stPos,edPos+1):
	    feat_vec = self.file_data[idx]
	    X_matrix[idx-stPos] = feat_vec
	# cross
	


	    
	    
	 

    

