from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import  LoadImages
from utils.datasets import LoadData
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class DetectImage:
    def __init__(self, img_size, conf_thres = 0.25, iou_thres = 0.45):
        base_path = str(Path().absolute())
        weights = base_path + "\\best.pt"
        self.imgsz = img_size
        self.conf_thres = 0.6
        self.iou_thres = 0.8
        set_logging() #enable console output
        self.device = select_device('') #select device cpu vs gpu
        self.model = attempt_load(weights, map_location=self.device)
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, img_size)


    def adjust_image_size(self, source):
        image = cv2.imread(source)
        if(image.shape[0] != self.imgsz or image.shape[1] != self.imgsz):
            self.h_diff = image.shape[0]/self.imgsz
            self.w_diff = image.shape[1]/self.imgsz
            #return cv2.resize(image, (self.imgsz, self.imgsz))
            cv2.imwrite(source, cv2.resize(image, (self.imgsz, self.imgsz)))

    def load_data(self, source):
        self.source = source
        #self.adjust_image_size(source)
        model = self.model
        stride = int(model.stride.max()) 
        self.dataset = LoadImages(source, img_size=self.imgsz, stride=stride)

    
    def get_face(self, image):
        model = self.model
        stride = int(model.stride.max()) 
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        faces = []
        
        im0s = image
        img = letterbox(im0s, self.imgsz, stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
     

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) #convert from 3 dimension to one

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)


        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)               
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    print(c1, c2)
                    cropped_image = im0[c1[1]:c2[1],c1[0]:c2[0]]
                    faces.append(cropped_image)

            return faces


    def get_face_from_file(self, source):
        self.load_data(source)
        model = self.model
        dataset = self.dataset

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        #path: filepath to current file being analyzed
        #img: image modified to used on detection
        #img0: original image
        #vid_cap: cv2 video capture object
        faces = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0) #convert from 3 dimension to one

            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process detections
          
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Pathcd 
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)               
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        print(c1, c2)
                        cropped_image = im0[c1[1]:c2[1],c1[0]:c2[0]]
                        faces.append(cropped_image)
                        
                       

            return faces
        


#detection = DetectImage(640)
#p = Path()
#source = p.absolute()
#image_src = str(p.joinpath(str(source), 'Fernando.jpg'))
#cv2.imshow('Fernando', detection.get_face(image_src))
#cv2.waitKey(0) 
