import os
import sys
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import datasets,transforms

from PIL import Image
from models.yolo_v2 import yolo_v2
from utils.cfg_loader import load_class_names
from utils.cfg_loader import read_data_cfg

from dataset_factory.VOCDataset import VOCDataset
from utils.nms import nms 
import cPickle
from utils.iou import bbox_iou
import time
import xml.etree.ElementTree as ET
def parse_anno(anno_file):
    tree = ET.parse(anno_file)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)

        bbox = obj.find('bndbox')
        obj_struct['bbox'] =[ int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)] 
        objects.append(obj_struct)
    return objects

def voc_eval(imageset_file,det_file,cls_name,cachedir,ovthresh=0.5):

    cachefile = os.path.join(cachedir,'voc_2007_test_annos.pkl')
    with open(imageset_file, 'r' ) as f:
        image_files = f.readlines()

    if not os.path.isfile(cachefile):
        recs = {}
        for i,imgname in enumerate(image_files):
            anno_name = imgname.replace('.jpg','.xml').replace('JPEGImages','Annotations')
            recs[imgname] = parse_anno(anno_name)
            if i %100 ==0:
                print 'Reading annotation for {:d}/{:d}'.format(i+1,len(image_files))
        with open(cachefile,'w') as f:
            cPickle.dump(recs,f)
    else:
        with open(cachefile,'r') as f:
            recs = cPickle.load(f)

    #load gt files
    gts = {}
    npos = 0
    for imgname in image_files:
        gt_per_img = [obj for obj in recs[imgname] if obj['name']==cls_name]
        bbox = np.array([obj['bbox'] for obj in gt_per_img])
        difficult = np.array([obj['difficult'] for obj in gt_per_img]).astype(np.bool)
        npos = npos + np.sum(~difficult)

        det = [False]*len(gt_per_img)
        gts[imgname] = {'bbox':bbox,
                        'difficult':difficult,
                        'det':det}
    print 'valid.py Positive objects %d in dataset'%(npos)
    #read dets
    if os.path.isfile(det_file):
        with open(det_file,'r') as f:
            lines = f.readlines()
        
        img_names = [line.strip().split(' ')[0] for line in lines]
        confidence =np.array([line.strip().split(' ')[1] for line in lines])
        detBndBoxes = np.array([[np.float(loc) for loc in line.strip().split(' ')[2:]] for line in lines])

        #sorted by confidence
        sorted_ind = np.argsort(-confidence)
        detBndBoxes = detBndBoxes[sorted_ind,:]
        img_names = [img_names[ind] for ind in sorted_ind]

        detnum = len(img_names)
        tp = np.zeros(detnum)
        fp = np.zeros(detnum)

        for detid in detnum:
            gts_for_img = gts[img_names[detid]]            
            det_bb      = detBndBoxes[detid]

            ovmax = -np.inf
            gt_bbs      = gts_for_img['bbox'].astype(np.float)

            if gt_bbs.size>0:
                print gt_bbs
                gt_bbs = gt_bbs.t()
                print gt_bbs
                overlaps = bbox_iou(gt_bbs,det_bb,x1y1x2y2=True)
                ovmax = np.max(overlaps)
                jmax  = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not gts_for_img['difficult'][jmax]:
                    if not gts_for_img['det'][jmax]:
                        tp[detid] = 1
                        gts_for_img['det'][jmax]= True
                    else:
                        fp[detid] = 1
            else:
                fp[detid] = 1
        fp = np.cumsum(fp)
        tp = tp.cumsum(tp)
        rec = tp/float(npos)

        prec = tp/(np.maximum(tp+fp,np.finfo(np.float64).eps))


        voc_2007_metric = True
        if voc_2007_metric:
            ap = 0.0 
            for t in np.arange(0.0,1.1,0.1):
                if np.sum(rec >=t)==0:
                    p = 0
                else:
                    p=np.max(prec[rec >=t]) 
                ap = ap + p/11.0
    else:
        print 'detfile %s not exist' %(det_file)    
        rec  = 0 
        prec = 0
        ap   = 0
    return rec,prec,ap
    pass




def valid(datacfg,weight_file,outfile_prefix):

    options = read_data_cfg(datacfg)
    valid_images_set_file = options['valid']
    namesfile = options['names']

    #load class names
    class_names = load_class_names(namesfile)
    #load valid image
    with open(valid_images_set_file,'r') as fp:
        image_files = fp.readlines()
        image_files = [file.rstrip() for file in image_files]


    model = yolo_v2()
    model.load_weights(weight_file)


    print("weights %s loaded"%(weight_file))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    #result file
    fps = [0]*model.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    dir_name = 'results/%s_%s_%s' %(namesfile.split('/')[-1].split('.')[0],weight_file.split('/')[-1].split('.')[0],time.strftime("%Y%m%d_%H%M%S",time.localtime()))
    print 'save results to %s'%(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in range(model.num_classes):
        buf ="%s/%s_%s.txt" % (dir_name,outfile_prefix,class_names[i])
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
            print "[Batch_index:%d] [%d/%d] file:%s "%(batch_index,LineId+1,len(image_files),image_name)

            img_orig = Image.open(image_name)
            #print img_orig
            height,width =img_orig.height,img_orig.width
            print "   height %d, width %d, bbox num %d" % (height,width,len(boxes))
            for box in boxes:
                x1 = (box[0] - box[2]/2.0)*width
                y1 = (box[1] - box[3]/2.0)*height
                x2 = (box[0] + box[2]/2.0)*width
                y2 = (box[1] + box[3]/2.0)*height
                det_conf = box[4]
                cls_id   = box[6]
                fps[cls_id].write("%s %f %f %f %f %f\n"%(image_name,det_conf,x1,y1,x2,y2))

    for i in range(model.num_classes):
        fps[i].close()

    pass
    cachedir       = 'results'
    imageset_file  = valid_images_set_file
    use_voc_metric = True

    avrPres        = []
    for cls_name in class_names:
        det_file = '%s_%s'%(outfile_prefix,cls_name)
        rec,prec,ap = voc_eval(imageset_file,det_file,cls_name,cachedir,ovthresh=0.5)
        avrPres.append(ap)
        print 'AP for {} = {:.4f}'.format(cls_name,ap)
    print 'Mean AP = {:.4f}'.format(np.mean(np.array(avrPres)))
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #get average precision using voc standard

if __name__=="__main__":
    if len(sys.argv) == 3:
        datacfg = sys.argv[1]
        weightfile = sys.argv[2]
        outfile = 'comp4_det_test'
        valid(datacfg,weightfile,outfile)
    else:
        print("Usage:")
        print("python valid.py datacfg weightfile")




