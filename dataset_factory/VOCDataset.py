import torch 
import numpy as np
import numpy.random as npr


from torch.utils.data import Dataset
from PIL import Image
def read_truth_args(lab_path, min_box_scale):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5,5)
        
        new_truths = []
        for i in range(truths.size[0]):
            if truths[i][3]<min_box_scale:
                continue
            new_truths.append(truths[i][0],truths[i][1],truths[i][2],truths[i][3],truths[i][5])
        return np.array(new_truths)
    else:
        return np.array([])

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
        if self.train_phase:
            pass

        else:
            img = Image.open(image_path).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)

            labelpath = image_path.replace('JPEGImages','labels').replace('.jpg','.txt').replace('.png','.txt')
            label = torch.zeros(50*5)

            try:
                tmp = torch.from_numpy(read_truth_args(labelpath,8.0/img.width).astype(no.float32))
            except Exception:
                tmp = torch.zeros(1,5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()

            if tsz >50*5:
                print ("warning labeled object morn than %d" %(50))
                label = tmp[0:50*5]
            else:
                label = tmp


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img,label)





