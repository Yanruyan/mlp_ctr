############################################
# desc : read data with mini-batch
# @author : qiangz2012@yeah.net
#
###########################################

from config import *
import numpy as np

class DataLoader(object):
    
    def __init__(self, file_path, data_type):
        self.file_path = file_path
	self.data_type = data_type
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
		if self.data_type == "Y":
                    features.append( float(segs[1]) )
		else:  # "X"
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
	nLines = 0
	for idx in range(stPos,edPos+1):
	    feat_vec = self.file_data[idx]
	    X_matrix[idx-stPos] = feat_vec
	    nLines += 1
	# cross
	if ed < stPos:
	    stPos = 0
	    edPos = ed
	    for idx in range(stPos,edPos+1):
	        feat_vec = self.file_data[idx]
		X_matrix[nLines] = feat_vec
	        nLines += 1
	return X_matrix


    def _test_data(self):
        print self.total
	idxs = [0,1,2,3,4,100,1000]
	for idx in idxs:
	    print self.file_data[idx]


