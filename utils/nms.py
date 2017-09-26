import torch 
from .iou import bbox_iou

def nms(boxes,nms_thresh):
	if len(boxes)==0:
		return boxes
	def_confs = torch.zeros(len(boxes))
	for i in range(len(boxes)):
		def_confs[i] = 1- boxes[i][4]

	_,sortIdx = torch.sort(def_confs)
	out_boxes = []
	for i in range(len(boxes)):
		box_i  = boxes[sortIdx[i]]
		if box_i[4] >0 :
			out_boxes.append(box_i)
			for j in range(i+1,len(boxes)):
				box_j = boxes[sortIdx[j]]
				if bbox_iou(box_i,box_j,x1y1x2y2=False) > nms_thresh:
					box_j[4] = 0
	return out_boxes