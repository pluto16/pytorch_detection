
import os
import sys
import time

from models.yolo_v2 import yolo_v2
from models.yolo_v2_resnet import yolo_v2_resnet
from models.yolo_v2_loss import yolo_v2_loss

from utils.cfg_loader import read_data_cfg
from utils.iou import bbox_iou
from utils.nms import nms
from dataset_factory.VOCDataset import VOCDataset

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from utils.logger import Logger

def file_lines(filepath):
    with open(filepath) as lines:
        return sum(1 for line in lines)

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

if __name__=="__main__":
    if len(sys.argv) == 3:
        datacfg = sys.argv[1]
        weightfile = sys.argv[2]
        backupdir = 'backup'

        #training settings
        data_options       = read_data_cfg(datacfg)
        train_set_file     = data_options['train']
        train_samples      = file_lines(train_set_file)
        valid_set_file     = data_options['valid']
        
        #training parameters
#       max_batches = 45000
        batch_size  = 40
        init_epoch    = 0
        max_epochs    = 100
        learning_rate = 0.0001
        momentum      = 0.9
        decay         = 0.0005
        seen_samples  = 0
        processed_batches = 0
        save_interval = 10
        steps         = [int(i) for i in "0,100,25000,35000".split(",") ]
        scales        = [float(i) for i in "0.1,10,0.1,0.1".split(",")]
# learning_rate=0.0001
# max_batches = 45000
# policy=steps
# steps=100,25000,35000
# scales=10,.1,.1

        #test parameters
        conf_thresh = 0.25
        nms_thresh  = 0.45
        iou_thresh  = 0.5
        ngpus       = 2 

        #logger for tensorboard
        log_dir = "./logs/log_%s" %(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tf_logger = Logger(log_dir)
        
        if not os.path.exists(backupdir):
            os.mkdir(backupdir)

        # construct model 
        #model = yolo_v2_resnet()
        old_model = yolo_v2()
        # to do load pretrained partial file
        old_model.load_weights(weightfile)
        if torch.cuda.is_available():
            #os.environ['CUDA_VISIBLE_DEVICES']='0'
            torch.cuda.manual_seed(int(time.time()))
            old_model.cuda()
            model = old_model
            if ngpus >1:
                model = nn.DataParallel(model,device_ids=[0,2])

            
        optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
        #scheduler = lr_scheduler.StepLR(optimizer,step_size = 30 ,gamma=0.1)
        def adjust_learning_rate(optimizer,batchid,learning_rate,batch_size):
            for i in range(len(steps)):
                if batchid == steps[i]:
                    learning_rate = learning_rate*scales[i]
            for param_group  in optimizer.param_groups:
                param_group['lr'] = learning_rate/batch_size
            return learning_rate

            
        #load train image set
        with open(train_set_file,'r') as fp:
            train_image_files = fp.readlines()
            train_image_files = [file.rstrip() for file in train_image_files]

        #load valid image set
        with open(valid_set_file,'r') as fp:
            valid_image_files = fp.readlines()
            valid_image_files = [file.rstrip() for file in valid_image_files]


        if old_model.network_name == 'yolo_v2':
            print 'img_trans has no transforms.Normalize'
            img_trans = transforms.Compose([transforms.ToTensor(),])

        elif old_model.network_name == 'yolo_v2_resnet':
            print 'img_trans has transforms.Normalize'
            img_trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        #construct valid data loader
        train_dataset = VOCDataset(train_image_files,shape=(old_model.width,old_model.height),shuffle=True,train_phase=True,transform=img_trans)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=batch_size,pin_memory=True)
        #logging('training with %d samples' % (len(train_loader.dataset)))
        #
        valid_dataset = VOCDataset(valid_image_files,shape=(old_model.width,old_model.height),shuffle=False,transform=img_trans)
        valid_loader  = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=batch_size,pin_memory=True)

        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i
        
        loss_func = yolo_v2_loss(old_model.num_classes,old_model.anchors_str,old_model.anchor_step)
        loss_func.tf_logger = tf_logger
        for epoch in range(init_epoch, max_epochs): 
            if 1:
                model.train()
                for batch_idx, (images, target) in enumerate(train_loader):
                    if processed_batches in steps:
                        
                        learning_rate = adjust_learning_rate(optimizer,processed_batches,learning_rate,batch_size)
                        print "learning rate changed to {}".format(learning_rate)
                    #scheduler.step()
                    #logging('epoch %d,all_batches %d, batch_size %d, lr %f' % (epoch+1,processed_batches+1,batch_size, learning_rate))
                    if torch.cuda.is_available():
                        images = images.cuda()
                    images_var = Variable(images)
                    target_var = Variable(target)
                    optimizer.zero_grad()
                    output = model(images_var)
                    seen_samples = seen_samples + images_var.data.size(0)
                    loss_func.seen  = seen_samples
                    loss_func.epoch = epoch
                    loss_func.lr           = learning_rate
                    loss_func.seenbatches = processed_batches
                    loss = loss_func(output, target_var)
                    
                    loss.backward()
                    optimizer.step()
                    processed_batches = processed_batches + 1

                if (epoch+1) % save_interval == 0:
                    extension = 'tmp'
                    if old_model.network_name == 'yolo_v2':
                        extension = 'weights'
                    elif old_model.network_name == 'yolo_v2_resnet':
                        extension = 'pth'
                    logging('save weights to %s/%06d.%s' % (backupdir, epoch+1,extension))
                    model.module.seen = (epoch + 1) * len(train_loader.dataset)
                    model.module.save_weights('%s/%06d.%s' % (backupdir, epoch+1,extension))

            #valid process 
            total       = 0.0
            proposals   = 0.0
            correct     = 0.0
            eps         = 1e-5
            model.eval()
            for batch_idx, (data, target) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                data = Variable(data, volatile=True)
                output = model(data).data
                all_boxes = old_model.get_region_boxes(output, conf_thresh)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)
                    total = total + num_gts
                    for i in range(len(boxes)):
                        if boxes[i][4]*boxes[i][5] > conf_thresh:
                            proposals = proposals+1
                    for i in range(num_gts):
                        box_gt   = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        for j in range(len(boxes)):
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > iou_thresh and boxes[j][6] == box_gt[6]:
                                correct = correct+1
            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("valid process precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
            tf_logger.scalar_summary("valid precision",precision,epoch)
            tf_logger.scalar_summary("valid recall",recall,epoch)
    else:
        print("Usage:")
        print("python train.py datacfg weightfile")