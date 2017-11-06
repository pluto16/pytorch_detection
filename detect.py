# detect.py
import sys
from PIL import Image,ImageDraw
from models.yolo_v2 import yolo_v2
import torch
from torch.autograd import Variable
import time

from utils.cfg_loader import load_class_names
from utils.iou import bbox_iou
from utils.nms import nms

def plot_boxes(img,boxes,savename=None,class_names=None):
	width  = img.width
	height = img.height

	draw = ImageDraw.Draw(img)
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = (box[0] - box[2]/2.0)*width
		y1 = (box[1] - box[3]/2.0)*height
		x2 = (box[0] + box[2]/2.0)*width
		y2 = (box[1] + box[3]/2.0)*height

		rgb = (255,0,0)
		if class_names:
			det_conf = box[4]
			cls_conf = box[5]
			cls_ind  = box[6]
			thr      = det_conf*cls_conf
			print ('%12s:cls_conf=%8.5f det_conf=%8.5f thr=%8.5f' %(class_names[cls_ind],cls_conf,det_conf,thr))
			rgb_anno = (0,0,255)
			draw.text((x1,y1),class_names[cls_ind],fill=rgb_anno)
		#print("{} {} {} {} ".format(x1,y1,x2,y2))
		draw.rectangle([x1,y1,x2,y2],outline=rgb)

	if savename:
		print("save plot results to {}".format(savename))
		img.save(savename)


def detect(namesfile, weightfile, imgfile):

	conf_thresh = 0.25
	nms_thresh  = 0.45
	model = yolo_v2()
	model.load_weights(weightfile)
	#model.save_weights('weights/save_test.weights')
	if torch.cuda.is_available():
		model.cuda()
	model.eval()

	img_orig = Image.open(imgfile).convert('RGB')
	siezd = img_orig.resize((model.width,model.height))

	if isinstance(siezd,Image.Image):
		img = torch.ByteTensor(torch.ByteStorage.from_buffer(siezd.tobytes()))
		img = img.view(model.height, model.width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, model.height, model.width)
        img = img.float().div(255.0)
	if torch.cuda.is_available():
		img =img.cuda()
	img =  Variable(img)

	start = time.time()
	output = model(img)
	output = output.data
	finish = time.time()
	boxes = model.get_region_boxes(output, conf_thresh)[0]

	#print("before nms")
	#print(boxes)
	boxes = nms(boxes, nms_thresh)
	#print("after nms")
	#print(boxes)
	print("{}: Predicted in {} seconds.".format(imgfile, (finish-start)))
	class_names = load_class_names(namesfile)
	plot_boxes(img_orig,boxes, 'predictions.jpg',class_names)


if __name__=='__main__':
	if len(sys.argv) == 4:
		namesfile  = sys.argv[1]
		weightfile = sys.argv[2]
		imgfile    = sys.argv[3]
		detect(namesfile,weightfile,imgfile)
	else:
		print("Usage: ")
		print("python detect.py namesfile weightfile imgfile")
		print("Please use yolo-voc.weights")
