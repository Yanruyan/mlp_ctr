########################################################
# desc : deep neural network model for CTR prediction.
# 
# @author : qiangz2012@yeah.net
#
########################################################

import tensorflow as tf
import numpy as np

class Dnn_Mlp(object):
  
    def __init__(self, epoch=10, batch_size=16,
                 lr=0.0005, optimizer="adam", l2_reg=0.01,
                 hidden_units=64, output_units=32, feat_size=7):
	self.epoch = epoch
	self.batch_size = batch_size
	self.lr = lr
	self.optimizer = optimizer
	self.l2_reg = l2_reg
	self.hidden_units = hidden_units
	self.output_units = output_units
	self.feat_size = feat_size

	self._init_graph()

    def _init_graph():
        self.graph = tf.Graph()
	with self.graph.as_default():  
	# have existing a default Graph in context, now we will
        # add some edges or nodes into it
            self.X = tf.placeholder(tf.float32,shape=[None,None])
	    self.Y = tf.placeholder(tf.float32,shape=[None,1]
	    
	    self.weights = self._init_weight()

            # ---------------- model ---------------------------
            # input -> hidden
	    # hidden -> output
	    # output -> label
	    # loss
	    # optimizer
	    # train

    def _init_weight(self):
        weights = dict()
        # W1: input -> hidden
	in_size = self.feat_size
	hidden_size = self.hidden_units
	glorot = np.sqrt( 2.0 / (in_size + hidden_size) )
        weights["layer_0"] = tf.Variable( np.random.normal(loc=0.0,scale=glorot,size=(in_size,hidden_size),dtype=np.float32) )
	weights["bias_0"] = tf.Variable( np.random.normal(loc=0.0,scale=glorot,size=(1,hidden_size),dtype=np.float32) )
	# W2: hiddent -> output
	out_size = self.output_units
	glorot = np.sqrt( 2.0 / (hidden_size + out_size) )
	weights["layer_1"] = tf.Variable( np.random.normal(loc=0.0,scale=glorot,size=(hidden_size,out_size),dtype=np.float32) )
	weights["bias_1"] = tf.Variable( np.random.normal(loc=0.0,scale=glorot,size=(1,out_size),dtype=np.float32) )
        # W3: concat
	label_size = 1
	glorot = np.sqrt( 2.0 / (out_size + label_size) )
	weights["layer_2"] = tf.Variable( np.random.normal(loc=0.0,scale=glorot,size=(out_size,1),dtype=np.float32) )
	weights["bias_2"] = tf.Variable( tf.constant(0.01), dtype=np.float32 )
	
	return weights
























		    









                                       
 	    
     

   





	
	
