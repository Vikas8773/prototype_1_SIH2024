# Apple-Detection using YOLOv3


YOLO is a pretrained network to detect 80 objectsfrom an image as mentioned in the.txt file.

# Files required for yolo

yolov3.txt - Contains the name of 80 objects

yolov3.cfg - contains the values of different layers like batch size, filters and more

# yolov3.weights - Pretrained Model

Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights)

# **Command format** 


 python yolo_opencv.py --image Apple\ Tree\ Images/apple-1.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt 



<img src="https://github.com/noorkhokhar99/Apple-Detection/blob/main/Screen%20Shot%201444-04-04%20at%207.09.33%20PM.png">
