import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
import sys
sys.path.insert(0, "..")
from utils.iou import bbox_iou

import math

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
		self.lr          = 0
		self.seenbatches = 0
		self.thresh      = 0.6
		self.tf_logger   = None
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
		conf_pred_cpu = self.convert2cpu(conf_pred.data)
		
		cls_preds  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
		cls_preds  = cls_preds.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

		#generate pred_bboxes
		pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
		grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
		anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
		anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		pred_boxes[0] = tx_pred.data.view(nB*nA*nH*nW) + grid_x
		pred_boxes[1] = ty_pred.data.view(nB*nA*nH*nW) + grid_y
		pred_boxes[2] = torch.exp(tw_pred.data.view(nB*nA*nH*nW)) * anchor_w
		pred_boxes[3] = torch.exp(th_pred.data.view(nB*nA*nH*nW)) * anchor_h
		pred_boxes = pred_boxes.transpose(0,1).contiguous().view(-1,4)
		pred_boxes_cpu = self.convert2cpu(pred_boxes)
		
		tx_target         	= torch.zeros(nB, nA, nH, nW) 
		ty_target         	= torch.zeros(nB, nA, nH, nW) 
		tw_target         	= torch.zeros(nB, nA, nH, nW) 
		th_target         	= torch.zeros(nB, nA, nH, nW) 
		coord_mask 			= torch.zeros(nB, nA, nH, nW)

		tconf_target      	= torch.zeros(nB, nA, nH, nW)
		conf_mask           = torch.ones(nB, nA, nH, nW)*self.noobject_scale
		tcls_target       	= torch.zeros(nB, nA, nH, nW) 
		cls_mask   			= torch.zeros(nB, nA, nH, nW)

		avg_anyobj = 0
		for b in xrange(nB):
			for j in xrange(nH):
				for i in xrange(nW):
					for n in xrange(nA):
						cur_pred_box = pred_boxes_cpu[b*nAnchors+n*nPixels+j*nW+i]
						best_iou = 0
						for t in xrange(50):
							if target[b][t*5+1] == 0:
								break
							gx = target[b][t*5+1]*nW
							gy = target[b][t*5+2]*nH
							gw = target[b][t*5+3]*nW
							gh = target[b][t*5+4]*nH
							cur_gt_box = np.array([gx,gy,gw,gh])
							iou = bbox_iou(cur_pred_box.numpy(), cur_gt_box, x1y1x2y2=False)
							if iou > best_iou:
								best_iou = iou

						if best_iou > self.thresh:
							conf_mask[b][n][j][i] = 0
						#avg_anyobj += conf_pred_cpu.data[b][n][j][i]

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
				pred_box = pred_boxes_cpu[b*nAnchors+best_n*nPixels+gj*nW+gi]
				#print "pred_box",pred_box

				tx_target[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
				ty_target[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
				tw_target[b][best_n][gj][gi] = math.log(target[b][t*5+3]* nW /self.anchors[self.anchor_step*best_n])
				th_target[b][best_n][gj][gi] = math.log(target[b][t*5+4]* nH /self.anchors[self.anchor_step*best_n+1])
				coord_mask[b][best_n][gj][gi] = self.coord_scale#self.coord_scale*(2-target[b][t*5+3]*target[b][t*5+4])
				
				iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
				tconf_target[b][best_n][gj][gi] = iou
				conf_mask[b][best_n][gj][gi] = self.object_scale

				cls_mask[b][best_n][gj][gi]    = 1
				tcls_target[b][best_n][gj][gi] = target[b][t*5]

				#print "b {} t {} iou {}".format(b,t,iou)
				if iou > 0.5:
					nCorrect = nCorrect + 1
				avg_iou += iou
				ncount +=1
				avg_obj += conf_pred_cpu[b][best_n][gj][gi]

		
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

		nProposals = int((conf_pred_cpu > 0.25).sum())
		print 'epoch: %d,seenB: %d,seenS: %d, nProposal %d, GtAvgIOU: %f, AvgObj: %f, AvgRecall: %f, count: %d'%(self.epoch,self.seenbatches,self.seen,nProposals,avg_iou/ncount,avg_obj/ncount,nCorrect*1.0/ncount,ncount)
		print('---->lr: %f,loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.lr,loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
		if self.tf_logger is not None:
			self.tf_logger.scalar_summary("loss_x", loss_x.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_y", loss_y.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_w", loss_w.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_h", loss_h.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_conf", loss_conf.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_cls", loss_cls.data[0], self.seenbatches)
			self.tf_logger.scalar_summary("loss_total", loss.data[0], self.seenbatches)


			self.tf_logger.scalar_summary("GtAvgIOU", avg_iou/ncount, self.seenbatches)
			self.tf_logger.scalar_summary("AvgObj", avg_obj/ncount, self.seenbatches)
			self.tf_logger.scalar_summary("AvgRecall", nCorrect*1.0/ncount, self.seenbatches)
			self.tf_logger.scalar_summary("count", ncount, self.seenbatches)


		return loss