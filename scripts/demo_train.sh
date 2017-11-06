
#tray yolo_v2
#python train.py data/voc.data weights/darknet19_448.conv.23
python train.py data/voc.data weights/000100.weights
#test valid part in train.py
#python train.py data/voc.data weights/yolo-voc.weights

#train yolo_v2_resnet
#python train.py data/voc.data weights/darknet19_448.conv.23.123