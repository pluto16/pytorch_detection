#
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from easydict import EasyDict as edict
import sys
sys.path.insert(0, "..")
from utils.iou import bbox_iou

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



class yolo_v2_loss(nn.Module):
	def __init__(self,num_classes,anchors_str,anchor_step):
		super(yolo_v2_loss,self).__init__()
		self.anchors_str = anchors_str
		self.anchors = [float(i) for i in anchors_str.split(',')]
		self.anchor_step = anchor_step
		self.num_classes = num_classes
		self.num_anchors = len(self.anchors)/anchor_step

		self.object_scale = 5  
		self.noobject_scale = 1 
		self.class_scale = 1
		self.coord_scale = 1
		self.seen        = 0
		self.epoch       = 0
		self.seenbatches = 0
		self.thresh      = 0.6
		self.mse_loss = nn.MSELoss(size_average=False)

	def convert2cpu(self,gpu_matrix):
		return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

	def forward( self, output, target):
		#output: 
		nB = output.data.size(0)
		nA = self.num_anchors
		nC = self.num_classes
		nH = output.data.size(2)
		nW = output.data.size(3)
		target = target.data
		nAnchors = nA*nH*nW
		nPixels  = nH*nW

		output   = output.view(nB, nA, (5+nC), nH, nW)

		tx_pred    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
		ty_pred    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
		tw_pred    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
		th_pred    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)

		conf_pred  = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))

		cls_preds  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
		cls_preds  = cls_preds.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

		#generate pred_bboxes
		pred_boxes = torch.FloatTensor(4, nB*nA*nH*nW)
		grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
		grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
		anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0]))
		anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1]))
		anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		pred_boxes[0] = tx_pred.cpu().data.view(nB*nA*nH*nW) + grid_x
		pred_boxes[1] = ty_pred.cpu().data.view(nB*nA*nH*nW) + grid_y
		pred_boxes[2] = torch.exp(tw_pred.cpu().data.view(nB*nA*nH*nW)) * anchor_w
		pred_boxes[3] = torch.exp(th_pred.cpu().data.view(nB*nA*nH*nW)) * anchor_h
		pred_boxes = pred_boxes.transpose(0,1).contiguous().view(-1,4)

		
		tx_target         	= torch.zeros(nB, nA, nH, nW) 
		ty_target         	= torch.zeros(nB, nA, nH, nW) 
		tw_target         	= torch.zeros(nB, nA, nH, nW) 
		th_target         	= torch.zeros(nB, nA, nH, nW) 
		coord_mask 			= torch.zeros(nB, nA, nH, nW)

		tconf_target      	= torch.zeros(nB, nA, nH, nW)
		conf_mask  			= torch.ones(nB, nA, nH, nW) * self.noobject_scale

		tcls_target       	= torch.zeros(nB, nA, nH, nW) 
		cls_mask   			= torch.zeros(nB, nA, nH, nW)

		avg_anyobj = 0
		for b in xrange(nB):
			for j in xrange(nH):
				for i in xrange(nW):
					for n in xrange(nA):
						cur_pred_box = pred_boxes[b*nAnchors+n*nPixels+j*nW+i]
						best_iou = 0
						for t in xrange(50):
							if target[b][t*5+1] == 0:
								break
							gx = target[b][t*5+1]*nW
							gy = target[b][t*5+2]*nH
							gw = target[b][t*5+3]*nW
							gh = target[b][t*5+4]*nH
							cur_gt_box = np.array([gx,gy,gw,gh])
							#print "cur_pred_boxes.size ",cur_pred_boxes.size()
							#print "cur_gt_boxes.size",cur_gt_boxes.size()
							iou = bbox_iou(cur_pred_box.numpy(), cur_gt_box, x1y1x2y2=False)
							if iou > best_iou:
								best_iou = iou
						if best_iou > self.thresh:
							conf_mask[b][n][j][i] = 0
						#avg_anyobj += conf_pred.cpu().data[b][n][j][i]

		if self.seen < 12800:
			tx_target.fill_(0.5)
			ty_target.fill_(0.5)
			tw_target.zero_()
			th_target.zero_()
			coord_mask.fill_(0.01)

		nGT = 0
		nCorrect = 0
		avg_iou= 0
		avg_obj= 0
		ncount = 0
		for b in xrange(nB):
			for t in xrange(50):
				if target[b][t*5+1] == 0:
					break
				nGT = nGT + 1
				best_iou = 0.0
				best_n = -1
				min_dist = 10000
				gx = target[b][t*5+1] * nW
				gy = target[b][t*5+2] * nH
				gi = int(gx)
				gj = int(gy)
				gw = target[b][t*5+3] * nW
				gh = target[b][t*5+4] * nH
				gt_box = [0, 0, gw, gh]
				for n in xrange(nA):
					aw = self.anchors[self.anchor_step*n]
					ah = self.anchors[self.anchor_step*n+1]
					anchor_box = [0, 0, aw, ah]
					iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
					if iou > best_iou:
						best_iou = iou
						best_n = n

				gt_box = [gx, gy, gw, gh]
				pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
				#print "pred_box",pred_box

				tx_target[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
				ty_target[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
				tw_target[b][best_n][gj][gi] = math.log(target[b][t*5+3]* nW /self.anchors[self.anchor_step*best_n])
				th_target[b][best_n][gj][gi] = math.log(target[b][t*5+4]* nH /self.anchors[self.anchor_step*best_n+1])
				coord_mask[b][best_n][gj][gi] = self.coord_scale#self.coord_scale*(2-target[b][t*5+3]*target[b][t*5+4])
				
				iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
				tconf_target[b][best_n][gj][gi] = iou
				conf_mask[b][best_n][gj][gi] = self.object_scale

				cls_mask[b][best_n][gj][gi] = 1
				tcls_target[b][best_n][gj][gi] = target[b][t*5]

				#print "b {} t {} iou {}".format(b,t,iou)
				if iou > 0.5:
					nCorrect = nCorrect + 1
				avg_iou += iou
				ncount +=1
				avg_obj += conf_pred.cpu().data[b][best_n][gj][gi]

		
		coord_mask_gpu = Variable(coord_mask.cuda())
		tx_target_gpu  = Variable(tx_target.cuda())
		ty_target_gpu  = Variable(ty_target.cuda())
		tw_target_gpu  = Variable(tw_target.cuda())
		th_target_gpu  = Variable(th_target.cuda())

		loss_x =  self.mse_loss(tx_pred*coord_mask_gpu, tx_target_gpu*coord_mask_gpu)/2.0
		loss_y =  self.mse_loss(ty_pred*coord_mask_gpu, ty_target_gpu*coord_mask_gpu)/2.0
		loss_w =  self.mse_loss(tw_pred*coord_mask_gpu, tw_target_gpu*coord_mask_gpu)/2.0
		loss_h =  self.mse_loss(th_pred*coord_mask_gpu, th_target_gpu*coord_mask_gpu)/2.0

		conf_mask_gpu = Variable(conf_mask.cuda())
		tconf_target_gpu = Variable(tconf_target.cuda())
		loss_conf = self.mse_loss(conf_pred*conf_mask_gpu, tconf_target_gpu*conf_mask_gpu)/2.0

		cls_mask = (cls_mask == 1)
		tcls_target_gpu  = Variable(tcls_target.view(-1)[cls_mask].long().cuda())
		cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
		cls_preds  = cls_preds[cls_mask].view(-1, nC)  
		loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls_preds, tcls_target_gpu)
		loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

		nProposals = int((conf_pred > 0.25).sum().data[0])
		print 'epoch %d, seen %d %d, Gt Avg IOU: %f, Obj: %f, Avg Recall: %f, count: %d'%(self.epoch,self.seenbatches,self.seen,avg_iou/ncount,avg_obj/ncount,nCorrect*1.0/ncount,ncount)
		print('     loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
		
		return loss


class yolo_v2(nn.Module):
	def __init__(self):
		super(yolo_v2,self).__init__()
		self.width    = 416
		self.height   = 416
		self.models,self.layerInd_has_no_weights   = self.create_model()
		self.header   = torch.IntTensor([0,0,0,0])
		self.seen     = 0
		self.anchors_str  = "1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071"
		self.num_classes      = 20
		self.anchor_step  = 2
		self.anchors      = [float(i) for i in self.anchors_str.lstrip().rstrip().split(',')]
		self.num_anchors  = len(self.anchors)/self.anchor_step
		self.network_name = "yolo_v2"
		
	def create_model(self):
		models = nn.ModuleList()
		layerInd_has_no_weights = []
		#32
		conv0  = nn.Sequential()
		conv0.add_module('conv0',nn.Conv2d(3,32,3,1,1,bias=False))
		conv0.add_module('bn0',nn.BatchNorm2d(32))
		conv0.add_module('leaky0',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv0)
		#max pool 0 ind=1
		models.append(nn.MaxPool2d(2,2))
		layerInd_has_no_weights.append(1)
		#64
		conv1  = nn.Sequential()
		conv1.add_module('conv1',nn.Conv2d(32,64,3,1,1,bias=False))
		conv1.add_module('bn1',nn.BatchNorm2d(64))
		conv1.add_module('leaky1',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv1)
		#max pool 1 ind=3
		models.append(nn.MaxPool2d(2,2))
		layerInd_has_no_weights.append(3)
        #128
		conv2 = nn.Sequential()
		conv2.add_module('conv2',nn.Conv2d(64,128,3,1,1,bias=False))
		conv2.add_module('bn2',nn.BatchNorm2d(128))
		conv2.add_module('leaky2',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv2)
		conv3 = nn.Sequential()
		conv3.add_module('conv3',nn.Conv2d(128,64,1,1,0,bias=False))
		conv3.add_module('bn3',nn.BatchNorm2d(64))
		conv3.add_module('leaky3',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv3)
		conv4 = nn.Sequential()
		conv4.add_module('conv4',nn.Conv2d(64,128,3,1,1,bias=False))
		conv4.add_module('bn4',nn.BatchNorm2d(128))
		conv4.add_module('leaky4',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv4)
		#max pool 2  ind =7
		models.append(nn.MaxPool2d(2,2))
		layerInd_has_no_weights.append(7)
		#256
		conv5 = nn.Sequential()
		conv5.add_module('conv5',nn.Conv2d(128,256,3,1,1,bias=False))
		conv5.add_module('bn5',nn.BatchNorm2d(256))
		conv5.add_module('leaky5',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv5)
		conv6 = nn.Sequential()
		conv6.add_module('conv6',nn.Conv2d(256,128,1,1,0,bias=False))
		conv6.add_module('bn6',nn.BatchNorm2d(128))
		conv6.add_module('leaky6',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv6)
		conv7 = nn.Sequential()
		conv7.add_module('conv7',nn.Conv2d(128,256,3,1,1,bias=False))
		conv7.add_module('bn7',nn.BatchNorm2d(256))
		conv7.add_module('leaky7',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv7)
		#max pool 3 ind=11
		models.append(nn.MaxPool2d(2,2))
		layerInd_has_no_weights.append(11)
		#512
		conv8 = nn.Sequential()
		conv8.add_module('conv8',nn.Conv2d(256,512,3,1,1,bias=False))
		conv8.add_module('bn8',nn.BatchNorm2d(512))
		conv8.add_module('leaky8',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv8)
		conv9 = nn.Sequential()
		conv9.add_module('conv9',nn.Conv2d(512,256,1,1,0,bias=False))
		conv9.add_module('bn9',nn.BatchNorm2d(256))
		conv9.add_module('leaky9',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv9)
		conv10 = nn.Sequential()
		conv10.add_module('conv10',nn.Conv2d(256,512,3,1,1,bias=False))
		conv10.add_module('bn10',nn.BatchNorm2d(512))
		conv10.add_module('leaky10',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv10)
		conv11 = nn.Sequential()
		conv11.add_module('conv11',nn.Conv2d(512,256,1,1,0,bias=False))
		conv11.add_module('bn11',nn.BatchNorm2d(256))
		conv11.add_module('leaky11',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv11)
		#keep result ind=16
		conv12 = nn.Sequential()
		conv12.add_module('conv12',nn.Conv2d(256,512,3,1,1,bias=False))
		conv12.add_module('bn12',nn.BatchNorm2d(512))
		conv12.add_module('leaky12',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv12)
		#max pool 4 ind=17
		models.append(nn.MaxPool2d(2,2))
		layerInd_has_no_weights.append(17)
		#1024
		conv13 = nn.Sequential()
		conv13.add_module('conv13',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv13.add_module('bn13',nn.BatchNorm2d(1024))
		conv13.add_module('leaky13',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv13)
		conv14 = nn.Sequential()
		conv14.add_module('conv14',nn.Conv2d(1024,512,1,1,0,bias=False))
		conv14.add_module('bn14',nn.BatchNorm2d(512))
		conv14.add_module('leaky14',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv14)
		conv15 = nn.Sequential()
		conv15.add_module('conv15',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv15.add_module('bn15',nn.BatchNorm2d(1024))
		conv15.add_module('leaky15',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv15)
		conv16 = nn.Sequential()
		conv16.add_module('conv16',nn.Conv2d(1024,512,1,1,0,bias=False))
		conv16.add_module('bn16',nn.BatchNorm2d(512))
		conv16.add_module('leaky16',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv16)
		conv17 = nn.Sequential()
		conv17.add_module('conv17',nn.Conv2d(512,1024,3,1,1,bias=False))
		conv17.add_module('bn17',nn.BatchNorm2d(1024))
		conv17.add_module('leaky17',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv17)
		##################################
		conv18 = nn.Sequential()
		conv18.add_module('conv18',nn.Conv2d(1024,1024,3,1,1,bias=False))
		conv18.add_module('bn18',nn.BatchNorm2d(1024))
		conv18.add_module('leaky18',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv18)

		#keep result id=24
		conv19 = nn.Sequential()
		conv19.add_module('conv19',nn.Conv2d(1024,1024,3,1,1,bias=False))
		conv19.add_module('bn19',nn.BatchNorm2d(1024))
		conv19.add_module('leaky19',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv19)

		#route -9 id=25 
		models.append(EmptyModule())
		layerInd_has_no_weights.append(25)
		#conv     id=26
		conv20 = nn.Sequential()
		conv20.add_module('conv20',nn.Conv2d(512,64,1,1,0,bias=False))
		conv20.add_module('bn20',nn.BatchNorm2d(64))
		conv20.add_module('leaky20',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv20)
		#reorg    id=27
		models.append(yolo_v2_reorg(2))
		layerInd_has_no_weights.append(27)
		#route -1,-4 id=28
		models.append(EmptyModule())
		layerInd_has_no_weights.append(28)
		

		#conv id =29
		conv21 = nn.Sequential()
		conv21.add_module('conv21',nn.Conv2d(1280,1024,3,1,1,bias=False))
		conv21.add_module('bn21',nn.BatchNorm2d(1024))
		conv21.add_module('leaky21',nn.LeakyReLU(0.1,inplace=True))
		models.append(conv21)

		conv22 = nn.Sequential()
		conv22.add_module('conv22',nn.Conv2d(1024,125,1,1,0))
		models.append(conv22)

		return models,layerInd_has_no_weights


	def get_region_boxes(self, output,conf_thresh):
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

		nnSoftmax = torch.nn.Softmax()

		cls_confs = nnSoftmax(Variable(output[5:5+num_classes].transpose(0,1))).data
		cls_max_confs,cls_max_ids = torch.max(cls_confs,1)
		cls_max_confs = cls_max_confs.view(-1)
		cls_max_ids   = cls_max_ids.view(-1)

		def_confs = self.convert2cpu(def_confs)
		cls_max_confs = self.convert2cpu(cls_max_confs)
		cls_max_ids   = self.convert2cpu_long(cls_max_ids)
		cx = self.convert2cpu(cx)
		cy = self.convert2cpu(cy)
		ws = self.convert2cpu(ws)
		hs = self.convert2cpu(hs)

		all_boxes = []
		for b in range(batch):
			boxes = []
			for row in range(h):
				for col in range(w):
					for i in range(num_anchors):
						ind = b*h*w*num_anchors + i*h*w + row*w + col
						conf  = def_confs[ind]*cls_max_confs[ind]
						if conf >conf_thresh:
							bcx = cx[ind]
							bcy = cy[ind]
							bw  = ws[ind]
							bh  = hs[ind] 
							#print "bbox {} {} {} {}".format(bcx,bcy,bw,bh)
							box = [bcx/w,bcy/h,bw/w,bh/h,def_confs[ind],cls_max_confs[ind],cls_max_ids[ind]] 
							boxes.append(box)
			all_boxes.append(boxes)
		return all_boxes

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
				if ind==16 or ind==27 or ind==24:
					outputs[ind]=x
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
		#cannot call .data on a torch.Tensor
		bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		#cannot call .data on a torch.Tensor
		bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]))
		start =start +num_b
		conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.size()))
		start = start + num_w
		return start


	def load_weights(self,weight_file):
		print "weight_file {}".format(weight_file)
		fp = open(weight_file,'rb')
		major = np.fromfile(fp,count=1,dtype = np.int32)
		minor = np.fromfile(fp,count=1,dtype = np.int32)
		revision = np.fromfile(fp,count=1,dtype = np.int32)
		print "weight_file major {} minor {}".format(major,minor)
		if (major[0]*10 + minor[0] )>=2:
			print "using version 2"
			self.seen = np.fromfile(fp,count=1,dtype = np.int64)
		else:
			print "using version 1"
			self.seen = np.fromfile(fp,count=1,dtype = np.int32)
		print "weight file revision {} seen {}".format(revision,self.seen)
		buf = np.fromfile(fp,dtype = np.float32)
		print "len(buf) = {} ".format(len(buf))
		fp.close()
		start = 0
		#print self.models
		for ind,model in enumerate(self.models):
			if ind not in self.layerInd_has_no_weights:
				if ind !=30:
					#print model[0]
					#print model[1]
					if start>=len(buf):
						continue
					start = self.load_conv_bn(buf, start, model[0], model[1])	
				else:
					if start>=len(buf):
						continue
					start = self.load_conv(buf,start,model[0])
	def save_weights(self,weight_file):
		fp = open(weight_file,'wb')
		header = np.asarray([0,0,0,self.seen],dtype=np.int32)
		header.tofile(fp)

		#save weights
		for ind,model in enumerate(self.models):
			if ind not in self.layerInd_has_no_weights:
				if ind !=30:
					self.save_conv_bn(fp,model[0],model[1])
				else:
					self.save_conv(fp,model[0])




	def convert2cpu(self,gpu_matrix):
		return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

	def convert2cpu_long(self,gpu_matrix):
		return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

	def save_conv(self,fp,conv_model):
		if conv_model.bias.is_cuda:
			self.convert2cpu(conv_model.bias.data).numpy().tofile(fp)
			self.convert2cpu(conv_model.weight.data).numpy().tofile(fp)
		else:
			conv_model.bias.data.numpy().tofile(fp)
			conv_model.weight.data.numpy().tofile(fp)

	def save_conv_bn(self,fp,conv_model,bn_model):
		if bn_model.bias.is_cuda:
			self.convert2cpu(bn_model.bias.data).numpy().tofile(fp)
			self.convert2cpu(bn_model.weight.data).numpy().tofile(fp)
			self.convert2cpu(bn_model.running_mean).numpy().tofile(fp)
			self.convert2cpu(bn_model.running_var).numpy().tofile(fp)
			self.convert2cpu(conv_model.weight.data).numpy().tofile(fp)		
		else:
			bn_model.bias.data.numpy().tofile(fp)
			bn_model.weight.data.numpy().tofile(fp)
			bn_model.running_mean.numpy().tofile(fp)
			bn_model.running_var.numpy().tofile(fp)
			conv_model.weight.data.numpy().tofile(fp)		






