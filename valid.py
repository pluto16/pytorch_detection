import os
import sys
import torch
from torch.autograd import Variable
from torchvision import datasets,transforms

from PIL import Image
from models.yolo_v2 import yolo_v2
from utils.cfg_loader import load_class_names
from utils.cfg_loader import read_data_cfg

from dataset_factory.VOCDataset import VOCDataset
from utils.nms import nms 




def valid(datacfg,weight_file,outfile):

    options = read_data_cfg(datacfg)
    valid_images_file = options['valid']
    namesfile = options['names']

    #load class names
    class_names = load_class_names(namesfile)
    #load valid image
    with open(valid_images_file,'r') as fp:
        image_files = fp.readlines()
        image_files = [file.rstrip() for file in image_files]


    model = yolo_v2()
    model.load_weights(weight_file)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    #result file
    fps = [0]*model.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(model.num_classes):
        buf ="%s/%s_%s.txt" % ('results',outfile,class_names[i])
        fps[i] = open(buf,'w')
    
    #construct datalist 
    valid_dataset = VOCDataset(image_files,shape=(model.width,model.height),shuffle=False,transform=transforms.Compose([transforms.ToTensor(),]))
    valid_loader  = torch.utils.data.DataLoader(valid_dataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)

    conf_thresh = 0.005
    nms_thresh  = 0.45
    LineId = -1
    for batch_index,(data,target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable( data, volatile= True)
        output = model(data).data
        batch_boxes = model.get_region_boxes(output,conf_thresh)
        for i in range(len(batch_boxes)):
            boxes = batch_boxes[i]
            boxes = nms(boxes,nms_thresh)

            LineId = LineId +1
            image_name = image_files[LineId]
            print "%d file:%s "%(LineId,image_name)

            size = Image.open(image_name)
            height,width =size[0],size[1] 
            print "   height %d, width %d"%(height,width)
            print "   bbox num %d" % (len(boxes))
            for box in boxes:
                x1 = (box[0] - box[2]/2.0)*width
                y1 = (box[1] - box[3]/2.0)*height
                x2 = (box[0] + box[2]/2.0)*width
                y2 = (box[1] + box[3]/2.0)*height
                det_conf = box[4]
                cls_id   = box[6]
                fps[cls_id].write("%s %f %f %f %f %f\n",image_name,det_conf,x1,y1,x2,y2)

    for i in range(model.num_classes):
        fps[i].close()

    pass
    #get average precision





if __name__=="__main__":
    if len(sys.argv) == 3:
        datacfg = sys.argv[1]
        weightfile = sys.argv[2]
        outfile = 'comp4_det_test_'
        valid(datacfg,weightfile,outfile)
    else:
        print("Usage:")
        print("python valid.py datacfg weightfile")




