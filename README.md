### pytorch_detection
Thanks to https://github.com/marvis/pytorch-yolo2
I try to reproduce the most popular cnn based object detection algorithm using pytorch 
This repository is trying to achieve the following goals
- [x] implement yolo v2 forward network using config yolo-voc.cfg
- [x] implement load darknet's  yolo-voc.weights()
- [x] implement detect function
- [o] implement valid function using pascal voc standard 
- [o] optimize forward process to make it using less gpu memory
- [o] implement darknet loss
- [o] implement training pascal voc