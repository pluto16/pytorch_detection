### pytorch_detection
This is a pytorch implementaion of YOLO v2 whichi attempts to reproduce the results of [project](https://pjreddie.com/darknet/yolo) and the [paper](https://arxiv.org/abs/1612.08242): YOLO9000: Better,Faster,Stronger by Joseph Redmon and Ali Farhadi.

This project is based on this [project](https://github.com/marvis/pytorch-yolo2)

This repository tries to achieve the following goals
- [x] implement yolo v2 forward network using config yolo-voc.cfg
- [x] implement load darknet's [yolo-voc.weights](http://pjreddie.com/media/files/yolo-voc.weights)
- [x] implement detect.py 
- [x] implement valid.py. This script produces results of pasval evaluation format for evaluation. 
- [x] implement eval.py. 
- [x] implement darknet loss
- [x] implement train.py. 
- [x] save as darknet weights
- [ ] support tensorboard 
- [ ] add image preprocess step to boost model accuracy

**NOTE:**
This is still an experimental project. Model trained on VOC0712 train+val
VOC07 test mAP is 0.5630 .@20171019 <br>
        AP for aeroplane       = 0.6112<br>
        AP for bicycle         = 0.6072<br>
        AP for bird            = 0.5375<br>
        AP for boat            = 0.4433<br>
        AP for bottle          = 0.3031<br>
        AP for bus             = 0.6043<br>
        AP for car             = 0.6910<br>
        AP for cat             = 0.6765<br>
        AP for chair           = 0.3610<br>
        AP for cow             = 0.5653<br>
        AP for diningtable     = 0.5538<br>
        AP for dog             = 0.6461<br>
        AP for horse           = 0.6997<br>
        AP for motorbike       = 0.6625<br>
        AP for person          = 0.6411<br>
        AP for pottedplant     = 0.3036<br>
        AP for sheep           = 0.5476<br>
        AP for sofa            = 0.5416<br>
        AP for train           = 0.6940<br>
        AP for tvmonitor       = 0.5701<br>

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