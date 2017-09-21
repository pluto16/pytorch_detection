# detect.py
import sys
from PIL import Image,ImageDraw
from models.yolo_v2 import yolo_v2
import torch
from torch.autograd import Variable

def load_class_names(namesfile):
	class_names = []
	with open(namesfile,'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.rstrip()
		class_names.append(line)
	return class_names
def bbox_iou(box1,box2, x1y1x2y2=True):
	if  x1y1x2y2:
		mx= min(box1[0],box2[0])
		Mx= max(box1[2],box2[2])
		my= min(box1[1],box2[1])
		My= max(box1[3],box2[3])
		w1= box1[2] - box1[0] 
		h1= box1[3] - box1[1]
		w2= box2[2] - box2[0]
		h2= box2[3] - box2[1]
	else:
		mx= min(box1[0]-box1[2]/2.0,box2[0]-box2[2]/2.0 )
		Mx= max(box1[0]+box1[2]/2.0,box2[0]+box2[2]/2.0 )
		my= min(box1[1]-box1[3]/2.0,box2[1]-box2[3]/2.0 )
		My= max(box1[1]+box1[3]/2.0,box2[1]+box2[3]/2.0 )
		w1= box1[2]
		h1= box1[3]
		w2= box2[2]
		h2= box2[3]
	uw = Mx-mx
	uh = My-my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	carea = 0
	if cw <= 0 or ch <=0:
		return 0.0
	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw*ch
	uarea = area1 + area2 - carea
	return carea/uarea
	
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
				if bbox_iou(box_i,box_j,x1y1x2y2=False) > nms_thresh
					box_j[4] = 0
	return out_boxes
def plot_boxes(img,boxes,savename=None,class_names=None):
	width = img.width
	height = img.height

	draw = ImageDraw.Draw(img)
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = (box[0] - box[2]/2.0)*width
		y1 = (box[1] - box[3]/2.0)*height
		x2 = (box[0] + box[2]/2.0)*width
		y2 = (box[1] - box[3]/2.0)*height

		rgb = (255,0,0)
		if class_names:
			thr      = box[4]
			cls_conf = box[5]
			cls_ind  = box[6]
			print ('%12s: %8.5f %8.5f' %(class_names[cls_ind],cls_conf,thr))
			rgb = (255,0,0)
			draw.text((x1,y1),class_names[cls_ind],fill=rgb)

		draw.rectangle([x1,y1,x2,y2],outline=rgb)
	if savename:
		print("save plot results to %s",savename)
		img.save(savename)


def detect(namesfile, weightfile, imgfile):


	conf_thresh = 0.25
	nms_thresh  = 0.4
	model = yolo_v2()
	model.load_weights(weightfile)

	if torch.cuda.is_available():
		model.cuda()
	model.eval()

	img = Image.open(imgfile).convert('RGB')
	sied = img.resize((model.width,model.height))

	

	if isinstance(img,Image.Image):
		img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
		img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
	if torch.cuda.is_available():
		img =img.cuda()
	img =  Variable(img)

	start = time.time()
	output = model(img)
	output = output.data
	finish = time.time()
	boxes = model.get_region_boxes(output, conf_thresh,nms_thresh)[0]
	boxes = nms(boxes, nms_thresh)
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

	class_names = load_class_names(namesfile)
	plot_boxes(img,boxes, 'predictions.jpg',class_names)


if __name__=='__main__':
	if len(sys.argv) == 3:
		weightfile = sys.argv[1]
		imgfile    = sys.argv[2]
		detect(weightfile,imgfile)
	else:
		print("Usage: ")
		print("python detect.py namesfile weightfile imgfile")