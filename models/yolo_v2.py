#

import torch
import torch.nn as nn
import torch.nn.functiional as F
from torch.autograd import Variable
import numpy as np
from easydict import EasyDict as edict


class EmptyModule(nn.Module):
	def __init__(self):
		super(EmptyModule,self).__init__()
	def forward(self,x):
		return x
class yolo_v2_reorg(nn.Module):
	def __init__(self,stride=2):
		super(yolo_v2_reorg,self).__init__()
		self.stride = stride
	def forward(self,x):
		stride = self.stride
		assert(x.data.dim()==4)
		B = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		assert(H % stride == 0)
		assert(W % stride == 0)
		ws = stride
		hs = stride
		x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        return x

class yolo_v2(nn.Module):
	def __init__(self):
		super(yolo_v2,self).__init__()
		self.width    = 416
		self.height   = 416
		self.models   = self.create_model()
		self.header   = torch.IntTensor([0,0,0,0])
		self.seen     = 0
		self.anchors_str  = "1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071"
		self.classes      = 20
		self.anchor_step  = 2
		self.anchors      = [float(i) for i in self.anchors_str.lstrip().rstrip().split(',')]
		self.num_anchors  = len(self.anchors)/self.anchor_step
		self.network_name = "yolo_v2"

		self.layerInd_has_no_weigts = []
	def create_model(self):
		models = nn.ModuleList()

		#32
		conv0  = nn.Sequential()
		conv0.add_moduel('conv0',nn.Conv2d(3,32,3,1,1,bias=False))
		conv0.add_moduel('bn0',nn.BatchNorm2d(32))
		conv0.add_moduel('leaky0',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv0)
		#max pool 0 ind=1
		models.append(nn.MaxPool2d(2,2))
		self.layerInd_has_no_weigts.append(1)
		#64
		conv1.add_moduel('conv1',nn.Conv2d(32,64,3,1,1,bias=False))
		conv1.add_moduel('bn1',nn.BatchNorm2d(64))
		conv1.add_moduel('leaky1',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv1)
		#max pool 1 ind=3
		models.append(nn.MaxPool2d(2,2))
		self.layerInd_has_no_weigts.append(3)
        #128
		conv2 = nn.Sequential()
		conv2.add_moduel('conv2',nn.Conv2d(64,128,3,1,1,bias=False))
		conv2.add_moduel('bn2',nn.BatchNorm2d(128))
		conv2.add_moduel('leaky2',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv2)
		conv3 = nn.Sequential()
		conv3.add_moduel('conv3',nn.Conv2d(128,64,1,1,0,bias=False))
		conv3.add_moduel('bn3',nn.BatchNorm2d(64))
		conv3.add_moduel('leaky3',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv3)
		conv4 = nn.Sequential()
		conv4.add_moduel('conv4',nn.Conv2d(64,128,3,1,1,bias=False))
		conv4.add_moduel('bn4',nn.BatchNorm2d(128))
		conv4.add_moduel('leaky4',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv4)
		#max pool 2  ind =7
		models.append(nn.MaxPool2d(2,2))
		self.layerInd_has_no_weigts.append(7)
		#256
		conv5 = nn.Sequential()
		conv5.add_moduel('conv5',nn.Conv2d(128,256,3,1,1,bias=False))
		conv5.add_moduel('bn5',nn.BatchNorm2d(256))
		conv5.add_moduel('leaky5',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv5)
		conv6 = nn.Sequential()
		conv6.add_moduel('conv6',nn.Conv2d(256,128,1,1,0,bias=False))
		conv6.add_moduel('bn6',nn.BatchNorm2d(128))
		conv6.add_moduel('leaky6',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv6)
		conv7 = nn.Sequential()
		conv7.add_moduel('conv7',nn.Conv2d(128,256,3,1,1,bias=False))
		conv7.add_moduel('bn7',nn.BatchNorm2d(256))
		conv7.add_moduel('leaky7',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv7)
		#max pool 3 ind=11
		models.append(nn.MaxPool2d(2,2))
		self.layerInd_has_no_weigts.append(11)
		#512
		conv8 = nn.Sequential()
		conv8.add_moduel('conv8',nn.Conv2d(256,512,3,1,1,bias=False))
		conv8.add_moduel('bn8',nn.BatchNorm2d(512))
		conv8.add_moduel('leaky8',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv8)
		conv9 = nn.Sequential()
		conv9.add_moduel('conv9',nn.Conv2d(512,256,1,1,0,bias=False))
		conv9.add_moduel('bn9',nn.BatchNorm2d(256))
		conv9.add_moduel('leaky9',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv9)
		conv10 = nn.Sequential()
		conv10.add_moduel('conv10',nn.Conv2d(256,512,3,1,1,bias=False))
		conv10.add_moduel('bn10',nn.BatchNorm2d(512))
		conv10.add_moduel('leaky10',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv10)
		conv11 = nn.Sequential()
		conv11.add_moduel('conv11',nn.Conv2d(512,256,1,1,0,bias=False))
		conv11.add_moduel('bn11',nn.BatchNorm2d(256))
		conv11.add_moduel('leaky11',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv11)
		#keep result ind=16
		conv12 = nn.Sequential()
		conv12.add_moduel('conv12',nn.Conv2d(256,512,3,1,1,bias=False))
		conv12.add_moduel('bn12',nn.BatchNorm2d(512))
		conv12.add_moduel('leaky12',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv12)
		#max pool 4 ind=17
		models.append(nn.MaxPool2d(2,2))
		self.layerInd_has_no_weigts.append(17)
		#1024
		conv13 = nn.Sequential()
		conv13.add_moduel('conv13',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv13.add_moduel('bn13',nn.BatchNorm2d(1024))
		conv13.add_moduel('leaky13',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv13)
		conv14 = nn.Sequential()
		conv14.add_moduel('conv14',nn.Conv2d(1024,512,1,1,0,bias=False))
		conv14.add_moduel('bn14',nn.BatchNorm2d(512))
		conv14.add_moduel('leaky14',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv14)
		conv15 = nn.Sequential()
		conv15.add_moduel('conv15',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv15.add_moduel('bn15',nn.BatchNorm2d(1024))
		conv15.add_moduel('leaky15',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv15)
		conv16 = nn.Sequential()
		conv16.add_moduel('conv16',nn.Conv2d(1024,512,1,1,0,bias=False))
		conv16.add_moduel('bn16',nn.BatchNorm2d(512))
		conv16.add_moduel('leaky16',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv16)
		conv17 = nn.Sequential()
		conv17.add_moduel('conv17',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv17.add_moduel('bn17',nn.BatchNorm2d(1024))
		conv17.add_moduel('leaky17',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv17)
		##################################
		conv18 = nn.Sequential()
		conv18.add_moduel('conv18',nn.Conv2d(1024,1024,3,1,1,bias=False))
		conv18.add_moduel('bn18',nn.BatchNorm2d(1024))
		conv18.add_moduel('leaky18',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv18)

		#keep result id=24
		conv19 = nn.Sequential()
		conv19.add_moduel('conv19',nn.Conv2d(1024,1024,3,1,1,bias=False))
		conv19.add_moduel('bn19',nn.BatchNorm2d(1024))
		conv19.add_moduel('leaky19',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv19)

		#route -9 id=25 
		models.append(EmptyModule())
		self.layerInd_has_no_weigts.append(25)
		#conv     id=26
		conv20 = nn.Sequential()
		conv20.add_moduel('conv20',nn.Conv2d(512,64,1,1,0,bias=False))
		conv20.add_moduel('bn20',nn.BatchNorm2d(64))
		conv20.add_moduel('leaky20',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv20)
		#reorg    id=27
		models.append(yolo_v2_reorg(2))
		self.layerInd_has_no_weigts.append(27)
		#route -1,-4 id=28
		models.append(EmptyModule())
		self.layerInd_has_no_weigts.append(28)
		

		#conv id =29
		conv21 = nn.Sequential()
		conv21.add_moduel('conv21',nn.Conv2d(1280,1024,3,1,1,bias=False))
		conv21.add_moduel('bn21',nn.BatchNorm2d(1024))
		conv21.add_moduel('leaky21',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv21)

		conv22 = nn.Sequential()
		conv22.add_moduel('conv22',nn.Conv2d(1024,125,1,1,0))
		models.append(conv22)

		return models


	def get_region_boxes(self, output,conf_thresh,nms_thresh):
		anchor_step = self.anchor_step
		num_classes = self.num_classes
		num_anchors = self.num_anchors
		anchors     = self.anchors
		if output.dim() ==3:
			output = output.unsequence(0)
		batch = output.size(0)
		assert(output.size(1) == (5+num_classes)*num_anchors)
		h = output.size(2)
		w = output.size(3)

		
		output = output.view(batch*num_anchors,5+num_classes,h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

		grid_x = torch.linspace(0,w-1,w).repeat(h,1).repeat(batch*num_anchors,1,1).view(batch*num_anchors*h*w).cuda()
		grid_y = torch.linspace(0,h-1,h).repeat(w,1).t().repeat(batch*num_anchors,1,1).view(batch*num_anchors*h*w).cuda()

		cx = torch.sigmoid(output[0]) + grid_x
		cy = torch.sigmoid(output[1]) + grid_y
		anchor_w = torch.Tensor(anchors).view(num_anchors,anchor_step).index_select(1,torch.LongTensor([0]))
		anchor_h = torch.Tensor(anchors).view(num_anchors,anchor_step).index_select(1,torch.LongTensor([1]))
		anchor_w = anchor_w.repeat(batch,1).repeat(1,1,h*w).view(batch*num_anchors*h*w).cuda()
		anchor_h = anchor_h.repeat(batch,1).repeat(1,1,h*w).view(batch*num_anchors*h*w).cuda()
		ws = torch.exp(output[2])*anchor_w
		hs = torch.exp(output[3])*anchor_h

		def_confs = torch.sigmoid(output[4])

		cls_confs = torch.nn.Softmax(Variable(output[5:5+num_anchors].transpose(0,1))).data
		cls_max_confs,cls_max_ids = torch.max(cls_confs,1)
		cls_max_confs = cls_max_confs.view(-1)
		cls_max_ids   = cls_max_ids.view(-1)

		def_confs = convert2cpu(def_confs)
		cls_max_confs = convert2cpu(cls_max_confs)
		cls_max_ids   = convert2cpu_long(cls_max_ids)
		cx = convert2cpu(cx)
		cy = convert2cpu(cy)
		ws = convert2cpu(ws)
		hs = convert2cpu(hs)

		all_boxes = []
		for b in range(batch):
			boxes = []
			for cy in range(h):
				for cx in range(w):
					for i in range(num_anchors):
						ind = b*h*w*num_anchors + i*h*w + cy*w +cx
						conf  = def_confs[ind]*cls_max_confs[ind]
						if conf >conf_thresh:
							box = [cx[ind]/w,cy[ind]/h,ws[ind]/w,ws[ind]/h,conf,cls_max_confs[ind],cls_max_ids[ind]] 
							boxes.append(box)
		all_boxes.append(boxes)






	def forward(self,x):
		outputs = dict()
		for ind,model in enumerate(self.models):
			#route 
			if ind == 25:
				input = outputs[ind-9]
				x = model(input)
			#route 
			elif ind == 28:
				input = torch.cat((outputs[ind-1],outputs[ind-4]),1)
				x = model(input)
			else:
				x = model(x)
				if id==16 or id==27 or id==24:
					outputs[id]=x
		return x

	def load_conv(self,buf,start,conv_model):
		num_w = conv_model.weight.numel()
		num_b = conv_model.bias.numel()
		conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))
		start = start +num_b
		conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.size()))
		start = start + num_w

		return start

	def load_conv_bn(self,buf,start,conv_model,bn_model):
		num_w = conv_model.weight.numel()
		num_b = bn_model.bias.numel()

		bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		bn_model.running_mean.data.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		bn_model.running_var.data.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.size()))
		start = start + num_w
		return start


	def load_weights(self,weight_file):
		fp = open(weight_file,'rb')
		major = np.fromfile(fp,count=1,dtype = np.int32)
		minor = np.fromfile(fp,count=1,dtype = np.int32)
		revision = np.fromfile(fp,coutn=1,dtype = np.int32)
		print "weight file major {} minor {}".format(major,minor)
		if (major[0]*10 + minor[0] )>=2:
			print "using version 2"
			seen = np.fromfile(fp,coutn=1,dtype = np.int64)
		else:
			print "using version 1"
			seen = np.fromfile(fp,coutn=1,dtype = np.int32)
		print "weight file revision {} seen {}".format(revision,seen)
		header = np.asarray([major,minor,revision,seen],dtype= np.int32)
		buf = np.fromfile(fp,dtype = np.float32)
		fp.close()
		starte = 0
		for ind,model in enumerate(self.models):
			if ind not in self.layerInd_has_no_weigts:
				if ind ==30:
					start = load_conv_bn(buf, start, model[0], model[1])	
				else:
					start = load_conv(buf,start,model[0])

	def convert2cpu(self,gpu_matrix):
		return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

	def convert2cpu_long(self,gpu_matrix):
		return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

	def save_conv(self,fp,conv_model):
		if conv_model.bias.is_cuda:
			convert2cpu(conv_model.bias.data).numpy().tofile(fp)
			convert2cpu(conv_model.weight.data).numpy().tofile(fp)
		else:
			conv_model.bias.data.numpy().tofile(fp)
			conv_model.weight.data.numpy().tofile(fp)
	def save_conv_bn(self,fp,conv_model,bn_model):
		if bn_model.bias.is_cuda:
			convert2cpu(bn_model.bias.data).numpy().tofile(fp)
			convert2cpu(bn_model.weight.data).numpy().tofile(fp)
			convert2cpu(bn_model.running_mean.data).numpy().tofile(fp)
			convert2cpu(bn_model.running_var.data).numpy().tofile(fp)
			convert2cpu(conv_model.weight.data).numpy().tofile(fp)		
		else:
			bn_model.bias.data.numpy().tofile(fp)
			bn_model.weight.data.numpy().tofile(fp)
			bn_model.running_mean.data.numpy().tofile(fp)
			bn_model.running_var.data.numpy().tofile(fp)
			conv_model.weight.data.numpy().tofile(fp)		






