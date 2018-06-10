########################################################
# desc : deep neural network model for CTR prediction.
# 
# @author : qiangz2012@yeah.net
#
########################################################

import tensorflow as tf
import numpy as np

class Dnn_Mlp(object):
  
    def __init__(self, epoch, batch_size, \
                 lr, optimizer_type, l2_reg, \
                 hidden_units, output_units, feat_size, \
		 activation, loss_type):
	self.epoch = epoch
	self.batch_size = batch_size
	self.lr = lr
	self.optimizer_type = optimizer_type
	self.l2_reg = l2_reg
	self.hidden_units = hidden_units
	self.output_units = output_units
	self.feat_size = feat_size
	self.activation = activation
	self.loss_type = loss_type

	self._init_graph()
	
#	init = tf.global_variables_initializer()

#	self.sess = self._init_session()
#	self.sess.run(init)

#	self.saver = tf.train.Saver()

	self.my_test()



    def my_test(self):
	print self.epoch                      
	print self.batch_size            
	print self.lr                            
	print self.optimizer_type    
	print self.l2_reg                   
	print self.hidden_units       
	print self.output_units        
	print self.feat_size              
	print self.activation           
	print self.loss_type              		


    def _init_graph(self):
        self.graph = tf.Graph()
	with self.graph.as_default():  
	# have existing a default Graph in context, now we will
        # add some edges or nodes into it
            self.X = tf.placeholder(tf.float32,shape=[None,self.feat_size])
	    self.Y = tf.placeholder(tf.float32,shape=[None,1])
	    
	    self.weights = self._init_weight()

            # ---------------- model ---------------------------
            # input -> hidden
	    hidden = self._add_neuron_layer(self.X,self.weights["layer_0"],self.weights["bias_0"],"hidden_layer",self.activation)    
	    # hidden -> output
	    output = self._add_neuron_layer(hidden,self.weights["layer_1"],self.weights["bias_1"],"output_layer",self.activation)
	    # output -> label
	    y = self._add_neuron_layer(output,self.weights["layer_2"],self.weights["bias_2"],"pCTR","sigmoid")

	    # --------------- train ---------------------------
	    # loss
	    if self.loss_type == "logloss":
		self.loss = tf.losses.log_loss(self.Y,y)
	    else: # mse
		self.loss = tf.losses.l2_loss( tf.subtract(self.Y,y) )
	    # L2
	    if self.l2_reg > 0.0:
		self.loss += tf.contrib.layers.l2_regularizer(tf.l2_reg)( self.weights["layer_2"] )
		self.loss += tf.contrib.layers.l2_regularizer(tf.l2_reg)( self.weights["layer_1"] )
		self.loss += tf.contrib.layers.l2_regularizer(tf.l2_reg)( self.weights["layer_0"] )
	    # optimizer
	    if self.optimizer_type == "adam":
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.9,beta2=0.999,epsilon=1e-8)
	    elif self.optimizer_type == "adagrad":
		self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr,initial_accumulator_value=1e-8)
	    else: # gd
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
	    # train
	    self.train_op = self.optimizer.minimize(self.loss)


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


    def _add_neuron_layer(self,X,W,b,name,activation=None):
        with tf.name_scope(name):
	    Z = tf.matmul(X,W)+b
	    if self.activation == "relu":
		return tf.nn.relu(Z)
	    elif self.activation == "sigmoid":
		return tf.sigmoid(Z)
	    elif self.activation == "tanh":
		return tf.tanh(Z)
	    else:
		return Z


    def _init_session(self):
        config_ = tf.ConfigProto(device_count={"gpu": 0})
	config_.gpu_options.allow_growth = True
	return tf.Session(config=config_)


    def _train(self, reader):
	# train MLP model
        for epoch in range(self.epoch):
	    X_batch,y_batch = reader._next_batch( self.batch_size )
	    loss,optim = self.sess.run( (self.loss, self.train_op), feed_dict={self.X:X_batch,self.Y:y_batch} )
	    print "epoch=%d,  loss=%f" % (epoch,loss)
	# save model
	self.saver.save(self.sess,"mlp_ctr.model")
	

























		    









                                       
 	    
     

   





	
	
