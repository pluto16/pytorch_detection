
import os
import sys
import xml.etree.ElementTree as ET
from utils.cfg_loader import load_class_names
from utils.cfg_loader import read_data_cfg
from utils.iou import bbox_iou
import cPickle
import numpy as np
import cPickle

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
        image_files = [file.rstrip() for file in image_files]

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
        gts_per_img = [obj for obj in recs[imgname] if obj['name']==cls_name]
        bboxes = np.array([agt['bbox'] for agt in gts_per_img])
        difficult = np.array([obj['difficult'] for obj in gts_per_img]).astype(np.bool)
        npos = npos + np.sum(~difficult)

        det = [False]*len(gts_per_img)
        gts[imgname] = {'bbs':bboxes,
                        'difficult':difficult,
                        'det':det}
    #print 'cls %s has %d gts in dataset'%(cls_name,npos)
    #read dets
    if os.path.isfile(det_file):
        with open(det_file,'r') as f:
            lines = f.readlines()
        
        img_names = [line.strip().split(' ')[0] for line in lines]
        confidence =np.array([float(line.strip().split(' ')[1]) for line in lines])
        detBndBoxes = np.array([[np.float(loc) for loc in line.strip().split(' ')[2:]] for line in lines])

        #sorted by confidence
        sorted_ind = np.argsort(-confidence)
        detBndBoxes = detBndBoxes[sorted_ind,:]
        img_names = [img_names[ind] for ind in sorted_ind]

        detnum = len(img_names)
        tp = np.zeros(detnum)
        fp = np.zeros(detnum)

        for detid in range(detnum):
            gts_for_img = gts[img_names[detid]]            
            det_bb      = detBndBoxes[detid]

            ovmax = -np.inf
            gt_bbs      = gts_for_img['bbs'].astype(np.float)

            if gt_bbs.size>0:
                #print gt_bbs
                gt_bbs = np.transpose(gt_bbs)
                #print gt_bbs
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
        tp = np.cumsum(tp)
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


if __name__=="__main__":
    if len(sys.argv) == 3:
        datacfg = sys.argv[1]
        result_dir = sys.argv[2]

        #read data cfg
        options = read_data_cfg(datacfg)
        valid_images_set_file = options['valid']
        namesfile = options['names']
        class_names = load_class_names(namesfile)
        resultfile_prefix = 'comp4_det_test'
        anno_cachedir       = 'anno_cached'
        if not os.path.exists(anno_cachedir):
            os.mkdir(anno_cachedir)

        avrPres        = []
        for cls_name in class_names:
            det_file = '%s/%s_%s.txt'%(result_dir,resultfile_prefix,cls_name)
            rec,prec,ap = voc_eval(valid_images_set_file,det_file,cls_name,anno_cachedir,ovthresh=0.5)
            avrPres.append(ap)
            print 'AP for {:15s} = {:.4f}'.format(cls_name,ap)
        print 'Mean AP = {:.4f}'.format(np.mean(np.array(avrPres)))
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    else:
        print("Usage:")
        print("python eval.py datacfg resultdir")