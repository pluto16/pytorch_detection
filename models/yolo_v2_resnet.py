
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class yolo_v2_resnet(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(yolo_v2_resnet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]      # delete the last fc layer and avrpool layer
        self.resnet = nn.Sequential(*modules)
        self.region_layer = nn.Conv2d(2048,125,1,1,0)
        self.width    = 416
        self.height   = 416
        self.header   = torch.IntTensor([0,0,0,0])
        self.seen     = 0
        self.anchors_str  = "1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071"
        self.num_classes   = 20
        self.anchor_step  = 2
        self.anchors      = [float(i) for i in self.anchors_str.lstrip().rstrip().split(',')]
        self.num_anchors  = len(self.anchors)/self.anchor_step
        self.network_name = "yolo_v2_resnet"

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = self.region_layer(features)
        return features

    def load_weights(self,weight_file):
        if os.path.isfile(weight_file,):
            print "load weight file: {}".format(weight_file)
            self.load_state_dict(torch.load(weight_file))
        else:
            print "weight file {} doesn't exist".format(weight_file)

    def save_weights(self,weight_dir):
        if os.path.exists(weight_dir):
            torch.save(self.state_dict(),weight_dir)

    def convert2cpu(self,gpu_matrix):
		return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def convert2cpu_long(self,gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


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