import tensorflow as tf
def batch_norm_layer(x,is_training,name=None):
	bn   =  tf.layers.batch_normalization(
        	inputs=x,
        	axis=-1,
        	momentum=0.05,
        	epsilon=0.00001,
	        center=True,
	        scale=True,
	        training = is_training
   	 )	
	return bn

