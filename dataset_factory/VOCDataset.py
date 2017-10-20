
import os
import torch 
import numpy as np
import numpy.random as npr


from torch.utils.data import Dataset
from PIL import Image
def read_truth_args(lab_path, min_box_scale):
    if os.path.exists(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5,5)
        #print "min_box_scale {} truths.size {}".format(min_box_scale,truths.size)
        new_truths = []
        for i in range(truths.shape[0]):
            if truths[i][3]<min_box_scale:
                continue
            new_truths.append([truths[i][0],truths[i][1],truths[i][2],truths[i][3],truths[i][4]])
        return np.array(new_truths)

def data_augmentation(img,shape,jitter,hue,saturation,exposure):

    sized = img.resize(shape)
    flip  = 0#npr.randint(1,10000)%2
    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    sx = 1.0
    sy = 1.0
    dx = 0.0
    dy = 0.0
    return sized,flip,dx,dy,sx,sy

def fill_truth_detection(labpath,w,h,flip,dx,dy,sx,sy):

    max_boxes_per_img = 50
    label = np.zeros((max_boxes_per_img,5))
    if os.path.exists(labpath):
        gts = np.loadtxt(labpath)
        if gts is not None:
            gts = np.reshape(gts,(-1,5))
            cc = 0
            for i in range(gts.shape[0]):
                # x1 = gts[i][1] - gts[i][3]/2
                # y1 = gts[i][2] - gts[i][4]/2
                # x2 = gts[i][1] + gts[i][3]/2
                # y2 = gts[i][2] + gts[i][4]/2

                # x1=min(0.999,max(0,x1*sx-dx))
                # y1=min(0.999,max(0,y1*sy-dy))
                # x2=min(0.999,max(0,x2*sx-dx))
                # y2=min(0.999,max(0,y2*sy-dy))

                # gts[i][1] = (x1+x2)/2
                # gts[i][2] = (y1+y2)/2
                # gts[i][3] = x2-x1
                # gts[i][4] = y2-y1

                if flip:
                    gts[i][1] = 0.999-gts[i][1]

                label[cc] =gts[i] 
                cc +=1
                if cc>=max_boxes_per_img:
                    break
    label = np.reshape(label,(-1))
    return label


class VOCDataset(Dataset):
    def __init__(self,image_files,shape=None,shuffle=True,batch_size=64,train_phase=False,transform=None,target_transform=None):
        super(VOCDataset,self).__init__()
        self.image_files = image_files
        if shuffle:
            npr.shuffle(self.image_files)

        self.image_num = len(self.image_files)
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.shape      = shape
        self.train_phase= train_phase
        
    def __len__(self):
        return self.image_num

    def __getitem__(self,index):

        image_path = self.image_files[index].rstrip()
        labelpath = image_path.replace('JPEGImages','labels').replace('.jpg','.txt').replace('.png','.txt')
        img = Image.open(image_path).convert('RGB')

        if self.train_phase:
            jitter = 0.3
            saturation = 1.5
            exposure = 1.5
            hue = 0.1
            img,flip,dx,dy,sx,sy = data_augmentation(img,self.shape,jitter,hue,saturation,exposure)
            label = fill_truth_detection(labelpath,img.width,img.height,flip,dx,dy,1./sx,1./sy)
            label = torch.from_numpy(label)
        else:
            if self.shape:
                img = img.resize(self.shape)
            label = torch.zeros(50*5)

            truths = read_truth_args(labelpath,8.0/img.width)

            
            #print "returned turthes {}".format(truths)
            tmp = torch.from_numpy(truths)

            tmp = tmp.view(-1)
            tsz = tmp.numel()

            if tsz >50*5:
                print ("warning labeled object morn than %d" %(50))
                label = tmp[0:50*5]
            else:
                label[0:tsz] = tmp


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img,label)





