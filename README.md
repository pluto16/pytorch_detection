### pytorch_detection
This is a pytorch implementaion of YOLO v2 whichi attempts to reproduce the results of [project](https://pjreddie.com/darknet/yolo) and the [paper](https://arxiv.org/abs/1612.08242): YOLO9000: Better,Faster,Stronger by Joseph Redmon and Ali Farhadi.

This project is based on this project [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)

This repository tries to achieve the following goals
- [x] implement yolo v2 forward network using config yolo-voc.cfg
- [x] implement load darknet's [yolo-voc.weights](http://pjreddie.com/media/files/yolo-voc.weights)
- [x] implement detect.py 
- [x] implement valid.py. This script produces results of pasval evaluation format for evaluation. 
- [x] implement eval.py. 
- [x] implement darknet loss
- [x] implement train.py. 
- [x] save as darknet weights
- [x] support log to tensorboard
- [x] support multi-gpu training  
- [x] add image preprocess step to boost model accuracy get 0.7303@20171106
- [ ] optimize code in yolo-v2 loss to reduce training time
**NOTE:**
This is still an experimental project. Model trained on VOC0712 train+val
VOC07 test mAP is 0.5630 @20171019 <br>
VOC07 test mAp is 0.7303 @20171106 <br>
        AP for aeroplane       = 0.784 <br>
        AP for bicycle         = 0.783 <br>
        AP for bird            = 0.754 <br>
        AP for boat            = 0.648 <br>
        AP for bottle          = 0.481 <br>
        AP for bus             = 0.777 <br>
        AP for car             = 0.824 <br>
        AP for cat             = 0.841 <br>
        AP for chair           = 0.56  <br>
        AP for cow             = 0.772 <br>
        AP for diningtable     = 0.719 <br>
        AP for dog             = 0.79  <br>
        AP for horse           = 0.807 <br>
        AP for motorbike       = 0.784 <br>
        AP for person          = 0.753 <br>
        AP for pottedplant     = 0.53  <br>
        AP for sheep           = 0.765 <br>
        AP for sofa            = 0.708 <br>
        AP for train           = 0.818 <br>
        AP for tvmonitor       = 0.709 <br>

### Detection Using a Pretrained Model
```
mkdir weights && cd weights
wget http://pjreddie.com/media/files/yolo-voc.weights
cd ..
./scripts/demo_detect.sh
```

### Training YOLOv2
You can train YOLOv2 on any dataset. Here we train on VOC2007/2012 train+val
1. Get the PASCAL VOC Data(2007trainval+2012trainval+2007test)
```
mkdir dataSet && cd dataSet
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
cd ..
```
2. Generate Labels for VOC
```
cd dataSet
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
3. Modify data/voc.data for Pascal Data
```
train  = dataSet/train.txt
valid  = dataSet/2007_test.txt
names = data/voc.names
backup = backup
```
4. Download Pretrained Convolutional Weights
```
cd weights
wget http://pjreddie.com/media/files/darknet19_448.conv.23
cd ..
```
5. Train The Model
```
./scripts/demo_train.sh
```
6. Evaluate The Model
if you want to eval the model, please modify the result directory in demo_eval.sh after running demo_valid 
```
./scripts/demo_valid.sh
./scripts/demo_eval.sh
```
