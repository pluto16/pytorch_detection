
import os
import sys
import time

from models.yolo_v2 import yolo_v2
from models.yolo_v2 import yolo_v2_loss

from utils.cfg_loader import read_data_cfg
from utils.iou import bbox_iou
from utils.nms import nms
from dataset_factory.VOCDataset import VOCDataset

import torch 
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms

def file_lines(filepath):
    with open(filepath) as lines:
        return sum(1 for line in lines)

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


def adjust_learning_rate(optimizer, lr, batch_id,steps,scales):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    assert len(steps)==len(scales)
    for i in range(len(steps)):
        scale = scales[i] 
        if batch_id == steps[i] and steps[0]>0 :
            lr = lr * scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            break

    return lr

if __name__=="__main__":
    if len(sys.argv) == 3:
        datacfg = sys.argv[1]
        weightfile = sys.argv[2]
        outdir = 'backup'

        #training settings
        data_options  = read_data_cfg(datacfg)
        train_set_file     = data_options['train']
        train_samples       = file_lines(train_set_file)
        valid_set_file      = data_options['valid']
        
        #training parameters
        max_batches = 80200
        batch_size  = 20
        init_epoch    = 0
        max_epochs    = max_batches*batch_size/train_samples+1
        learning_rate = 0.001/batch_size
        momentum      = 0.9
        decay         = 0.0005
        steps         = [int(step) for step in '-1,500,40000,60000'.split(',')]
        scales        = [float(scale) for scale in '0.1,10,.1,.1'.split(',')]
        seen_samples  = 0
        processed_batches = 0
        save_interval = 10
        #test parameters
        conf_thresh = 0.25
        nms_thresh  = 0.45
        iou_thresh  = 0.5

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # construct model 
        model = yolo_v2()
        # to do load pretrained partial file
        model.load_weights(weightfile)
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES']='0'
            torch.cuda.manual_seed(int(time.time()))
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

        #load train image set
        with open(train_set_file,'r') as fp:
            train_image_files = fp.readlines()
            train_image_files = [file.rstrip() for file in train_image_files]

        #load valid image set
        with open(valid_set_file,'r') as fp:
            valid_image_files = fp.readlines()
            valid_image_files = [file.rstrip() for file in valid_image_files]

        #construct valid datalist 
        valid_dataset = VOCDataset(valid_image_files,shape=(model.width,model.height),shuffle=False,transform=transforms.Compose([transforms.ToTensor(),]))
        valid_loader  = torch.utils.data.DataLoader(valid_dataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)

        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i
        
        loss_func = yolo_v2_loss(model.num_classes,model.anchors_str,model.anchor_step)

        for epoch in range(init_epoch, max_epochs): 
            #train process
            # train_dataset = VOCDataset(train_image_files,shape=(model.width,model.height),shuffle=True,train_phase=True,transform=transforms.Compose([transforms.ToTensor(),]))
            # train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=False,num_workers=batch_size,pin_memory=True)
            # logging('training with %d samples' % (len(train_loader.dataset)))

            # model.train()
            # for batch_idx, (data, target) in enumerate(train_loader):
            #     learning_rate = adjust_learning_rate(optimizer,learning_rate,processed_batches,steps,scales)
                
            #     #logging('epoch %d,all_batches %d, batch_size %d, lr %f' % (epoch+1,processed_batches+1,batch_size, learning_rate))

            #     if torch.cuda.is_available():
            #         data = data.cuda()
            #     data, target = Variable(data), Variable(target)
            #     optimizer.zero_grad()
            #     output = model(data)
            #     seen_samples = seen_samples + data.data.size(0)
            #     loss_func.seen  = seen_samples
            #     loss_func.epoch = epoch
            #     loss_func.seenbatches = processed_batches
            #     loss = loss_func(output, target)
                
            #     loss.backward()
            #     optimizer.step()
            #     processed_batches = processed_batches + 1

            # if (epoch+1) % save_interval == 0:
            #     logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
            #     model.seen = (epoch + 1) * len(train_loader.dataset)
            #     model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

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
                all_boxes = model.get_region_boxes(output, conf_thresh)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)
                    total = total + num_gts

                    for i in range(len(boxes)):
                        if boxes[i][4] > conf_thresh:
                            proposals = proposals+1

                    for i in range(num_gts):
                        box_gt   = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        for j in range(len(boxes)):
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        #print "best_iou {}  len(boxes) {} ".format(best_iou, len(boxes))
                            if iou > iou_thresh and boxes[j][6] == box_gt[6]:
                                correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("valid process precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    else:
        print("Usage:")
        print("python train.py datacfg weightfile")