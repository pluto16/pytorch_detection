#

import torch
import torch.nn as nn
import torch.nn.functiional as F
import numpy as np
from easydict import EasyDict as edict

class yolo_v2(nn.Module):
	def __init__(self):
		super(yolo_v2,self).__init__()
		self.blocks = self.create_blocks()
		self.models  = self.create_model()
		self.header = torch.IntTensor([0,0,0,0])
		self.seen   = 0


	def create_model(self):
		models = nn.ModuleList()
		conv0  = nn.Sequential()
		conv0.add_moduel('conv0',nn.Conv2d(3,32,3,1,1,bias=False))
		conv0.add_moduel('bn0',nn.BatchNorm2d(32))
		conv0.add_moduel('leaky0',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv0)

		pool0  = nn.Sequential()
		pool0.add_moduel('mxpool1',nn.MaxPool2d(2,2))
		models.append(pool0)

		conv1.add_moduel('conv1',nn.Conv2d(32,64,3,1,1,bias=False))
		conv1.add_moduel('bn1',nn.BatchNorm2d(64))
		conv1.add_moduel('leaky1',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv1)


	def forward(self,x):
		for block in blocks:


	def load_weights(self,weight_file):


	def create_blocks(self):
		blocks = []

		block = edict()
		block.type    		= "convolutional"
		block.filters 		= 32
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block = edict()
		block.size         = 2
		block.stride       = 2
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 64
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block = edict()
		block.size         = 2
		block.stride       = 2
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 128
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 64
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 128
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block = edict()
		block.size         = 2
		block.stride       = 2
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 256
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 128
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 256
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block = edict()
		block.size         = 2
		block.stride       = 2
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 512
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 256
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 512
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 256
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 512
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)


		block = edict()
		block.size         = 2
		block.stride       = 2
		blocks.append(block)


		block.type    		= "convolutional"
		block.filters 		= 1024
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 512
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 1024
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 512
		block.size    		= 1
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 1024
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)
##################################
		block.type    		= "convolutional"
		block.filters 		= 1024
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "convolutional"
		block.filters 		= 1024
		block.size    		= 3
		block.pad     		= 1
		block.stride  		= 1
		block.activation 	= "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		= "route"
		block.layers 		= "-9"
		blocks.append(block)

		block.type    		  = "convolutional"
		block.filters 		  = 64
		block.size    		  = 1
		block.pad     		  = 1
		block.stride  		  = 1
		block.activation 	  = "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		  = "route"
		block.layers 		  = "-1,-4"
		blocks.append(block)

		block.type    		  = "convolutional"
		block.filters 		  = 1024
		block.size    		  = 3
		block.pad     		  = 1
		block.stride  		  = 1
		block.activation 	  = "leaky"
		block.batch_normalize = 1
		blocks.append(block)

		block.type    		  = "convolutional"
		block.filters 		  = 125
		block.size    		  = 1
		block.pad     		  = 1
		block.stride  		  = 1
		block.activation 	  = "linear"
		blocks.append(block)

		block.type    		  = "region"
		block.anchors 		  = "1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071"
		block.bias_match      = 1
		block.classes     	  = 20
		block.coords  		  = 4
		block.num             = 5
		block.softmax         = 1
		block.jitter          = 0.3
		block.rescore         = 1
		block.object_scale    = 5
		block.nnobject_scale  = 1
		block.class_scale     = 1
		block.coord_scale     = 1
		block.absolute        = 1
		block.thresh          = 0.6
		block.random          = 1
		
		blocks.append(block)




		return blocks





