# implement darknet detector
from __future__ import division
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt

from bbox import bbox_iou
from darknet import Darknet
from util import load_classes

class Darknet_Detector():
    def __init__(self, cfg_file,wt_file,class_file, nms_threshold = .5, conf = 0.5, resolution=1024, num_classes=80, nms_classwise= True):
        #Set up the neural network
        print("Loading network.....")
        self.model = Darknet(cfg_file)
        self.model.load_weights(wt_file)
        print("Network successfully loaded")
        
        self.nms = nms_threshold
        self.conf = conf
        self.nms_classwise = nms_classwise
        self.resolution = resolution # sets size of max dimension
        
        self.CUDA = torch.cuda.is_available()
        
        self.num_classes = num_classes
        self.classes = load_classes(class_file) 
    

        
        self.model.net_info["height"] = self.resolution
        inp_dim = int(self.model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
    
        #If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()
        
        
        #Set the model in evaluation mode
        self.model.eval()
        
        
        
        
    #def detect(im_file, show = True, save = True, verbose = True):
        
net = Darknet_Detector('cfg/yolov3.cfg','yolov3.weights','data/coco.names')
