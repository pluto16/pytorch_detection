
import os
import torch 
import numpy as np
import numpy.random as npr


from torch.utils.data import Dataset
from PIL import Image,ImageDraw
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

def rand_scale(s):
    scale = npr.uniform(1,s)
    if (npr.randint(0,10000)%2):
        return scale
    return 1./scale

def distort_image(im,fhue,fsat,fexp):
    im = im.convert("HSV")
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i*fsat)
    cs[2] = cs[2].point(lambda i: i*fexp)

    def change_hue(x):
        x += fhue*255
        if x>255:
            x-=255
        if x<=0:
            x+=255
        return x
    cs[0] =cs[0].point(change_hue)
    im = Image.merge(im.mode,tuple(cs))
    im = im.convert('RGB')
    return im 

def random_distort_image(img,hue,saturation,exposure):
    fhue = npr.uniform(-hue,hue)
    fsat = rand_scale(saturation)
    fexp = rand_scale(exposure)
    res = distort_image(img,fhue,fsat,fexp)
    return res

def data_augmentation(img,shape,jitter,hue,saturation,exposure):

    w = shape[0]
    h = shape[1]
    ow = img.width
    oh = img.height
    dw = int(jitter*ow)
    dh = int(jitter*oh)

    new_ar = (ow- npr.randint(-dw,dw))*1.0/(oh - npr.randint(-dh,dh))
    scale  = npr.random()*(2-0.25)+0.25

    if new_ar<1:
        nh = scale*h 
        nw = nh*new_ar
    else:
        nw = scale*w
        nh = nw/new_ar
    nh = int(nh)
    nw = int(nw)
    nw_im = img.resize((nw,nh))
    out_im=Image.new(img.mode,shape,(128,128,128))
    nw_im_np = np.array(nw_im)
    out_im_np = np.array(out_im)
    
    dx = npr.random()*abs(w-nw) + min(0, w-nw)
    dy = npr.random()*abs(h-nh) + min(0, h-nh)

    dx = int(dx)
    dy = int(dy)
    # print "dx %d dy %d" %(dx,dy)
    # print nw_im_np.shape
    if dx <0:
        nw_im_start_x  = abs(dx)
        nw_im_end_x    = min(abs(dx)+w,nw)
        out_im_start_x = 0
        out_im_end_x   = min(abs(dx)+w,nw)-abs(dx)
    else:
        nw_im_start_x  = 0 
        nw_im_end_x    = min(nw,w-dx)
        out_im_start_x = dx
        out_im_end_x   = min(w,dx+min(nw,w-dx))
    if dy <0:
        nw_im_start_y  = abs(dy) 
        nw_im_end_y    = min(abs(dy)+h,nh)
        out_im_start_y = 0
        out_im_end_y   = min(abs(dy)+h,nh)-abs(dy)
    else:
        nw_im_start_y  = 0
        nw_im_end_y    = min(nh,h-dy)
        out_im_start_y = dy
        out_im_end_y   = min(h,dy+min(nh,h-dy))

    out_im_np[out_im_start_y:out_im_end_y,out_im_start_x:out_im_end_x,:] = nw_im_np[nw_im_start_y:nw_im_end_y,nw_im_start_x:nw_im_end_x,:]
    out_im = Image.fromarray(out_im_np)

    dx = -dx*1.0/w
    dy = -dy*1.0/h

    sx = nw*1.0/w
    sy = nh*1.0/h


    out_im = random_distort_image(out_im,hue,saturation,exposure)

    flip  = npr.randint(1,10000)%2
    if flip:
        out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)
    return out_im,flip,dx,dy,sx,sy

def fill_truth_detection(labpath,w,h,flip,dx,dy,sx,sy):

    max_boxes_per_img = 30
    label = np.zeros((max_boxes_per_img,5))
    if os.path.exists(labpath):
        gts = np.loadtxt(labpath)
        if gts is not None:
            
            #print gts
            gts = np.reshape(gts,(-1,5))
            npr.shuffle(gts)
            cc = 0
            for i in range(gts.shape[0]):
                x1 = gts[i][1] - gts[i][3]/2
                y1 = gts[i][2] - gts[i][4]/2
                x2 = gts[i][1] + gts[i][3]/2
                y2 = gts[i][2] + gts[i][4]/2

                x1=min(0.999,max(0,x1*sx-dx))
                y1=min(0.999,max(0,y1*sy-dy))
                x2=min(0.999,max(0,x2*sx-dx))
                y2=min(0.999,max(0,y2*sy-dy))

                gts[i][1] = (x1+x2)/2
                gts[i][2] = (y1+y2)/2
                gts[i][3] = x2-x1
                gts[i][4] = y2-y1

                if flip:
                    gts[i][1] = 0.999-gts[i][1]
                if gts[i][3]<0.002 or gts[i][4]<0.002:
                    continue
                label[cc] =gts[i] 
                cc +=1
                if cc>=max_boxes_per_img:
                    break
    else:
        print "label path not exist!"
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
            # print labelpath
            label = fill_truth_detection(labelpath,img.width,img.height,flip,dx,dy,sx,sy)
            # out_im_draw = ImageDraw.Draw(img)
            # label = np.reshape(label,(-1,5))
            # print label.shape
            # print label
            # for i in range(label.shape[0]):
            #     if label[i][1] ==0 :
            #         continue
            #     cx = label[i][1]*img.width
            #     cy = label[i][2]*img.height
            #     w =  label[i][3]*img.width
            #     h = label[i][4]*img.height
            #     new_loc = [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
            #     out_im_draw.rectangle(new_loc,outline=(0,0,255))
            # img.save('load_test_1.PNG','PNG')
            # label = np.reshape(label, (-1))
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





