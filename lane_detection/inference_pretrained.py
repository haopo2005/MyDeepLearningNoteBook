import os
import sys
import random
import logging
import argparse
import subprocess
from time import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from lib.config import Config
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
import torch.nn as nn
from collections import OrderedDict

class JSTNET(nn.Module):
    def __init__(self, model_elas_cls, extra_outputs):
        super(JSTNET, self).__init__()
        self.model = model_elas_cls
        # self.model.model._fc.regular_outputs_layer.in_features --> 1280
        self.model.model._fc.regular_outputs_layer = nn.Linear( self.model.model._fc.regular_outputs_layer.in_features, 70)
        self.model.model._fc.extra_outputs_layer = nn.Linear( self.model.model._fc.regular_outputs_layer.in_features, 30)
        self.share_top_y = True
        self.curriculum_steps = [0, 0, 0, 0]
        self.pred_category = True
        self.extra_outputs = 30
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        output, extra_outputs = self.model(inputs)
        return output, extra_outputs
    
    def decode(self, all_outputs, batch_size, conf_threshold=0.5):
        outputs, extra_outputs = all_outputs
        if extra_outputs is not None:
            extra_outputs = extra_outputs.reshape(batch_size, 10, -1)
            #extra_outputs = extra_outputs.argmax(dim=2)
        outputs = outputs.reshape(len(outputs), -1, 7)  # score + upper + lower + 4 coeffs = 7
        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])
        #outputs[outputs[:, :, 0] < conf_threshold] = 0 #fake lane will be filtered here
        result_outputs = outputs[outputs[:, :, 0] > conf_threshold]
        result_extra_outputs = extra_outputs[outputs[:, :, 0] > conf_threshold]

        if False and self.share_top_y:
            outputs[:, :, 0] = outputs[:, 0, 0].expand(outputs.shape[0], outputs.shape[1])

        #return outputs, extra_outputs
        #print(result_outputs.shape, result_extra_outputs.shape)
        return result_outputs, result_extra_outputs

# extract points for single lane and lane conf
def pred2lanes(pred, upper_shared, img_h, img_w):
    lanes = []
    # score + upper + lower + 4 coeffs = 7
    lane = pred[1:]  # remove conf
    lower, upper = lane[0], lane[1] #max_y, min_y
    lane = lane[2:]  # remove upper, lower positions --> polynomial coefficients
    
    # generate points from the polynomial
    #print(lower, upper, upper_shared)
    ys = np.linspace(upper, lower, num=10)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(int)
    points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)].tolist()
    
    #filter bad/repeat points
    for i in range(len(points)-1):
        temp = points[i]
        temp2 = points[i+1]
        if temp[0] == temp2[0] and temp[1] == temp2[1]:
            print('find repeated points')
            points = []
            return points, pred[0]

    return points, pred[0]

def generate_lines_per_img(predictions):
    lines_per_batch = []
    conf_batch = []
    upper_fist_lane = predictions[0][2]
    for idx in range(len(predictions)):
        points, conf = pred2lanes(predictions[idx], upper_fist_lane, 720, 1280) #points for each image
        lines_per_batch.append(points)
        conf_batch.append(conf)
    return lines_per_batch, conf_batch


cls_dict = {'1': 'solid', '2': 'dashed', '3': 'unknown'}

def test(model, test_loader, cfg):
    logging.info("Starting testing.")
    epoch = 1238 #trained epoches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join('/home/jns2szh/code/PolyLaneNet-master/experiments/bosch_3_class', "models", "model_{:03d}.pt".format(epoch)))['model'])
    model.eval()

    test_parameters = cfg.get_test_parameters()
    with torch.no_grad():
        test_img_list = pd.read_csv('/home/jns2szh/code/PolyLaneNet-master/test.csv')
        img = []
        index = []
        img_width = []
        img_height = []
        lane_prob = []
        solid_prob = []
        points = []
        solid_type = []

        #tips: only one image each time
        idx = 0
        for image, image_path in test_loader:
            image = image.to(device)
            outputs = model(image)
            outputs, extra_outputs = model.decode(outputs, 1, **test_parameters) #assume batch size is 1
            lane_cls = F.softmax(extra_outputs, dim=1)
            #print(lane_cls.cuda().data.cpu().numpy())
            #print(outputs.cuda().data.cpu().numpy())
            lines_per_img, lane_prob_per_img = generate_lines_per_img(outputs.cuda().data.cpu().numpy())

            solid_prob_img = []
            solid_type_img = []
            result = lane_cls.cuda().data.cpu().numpy() # 1*10*2
            for i in range(len(result)):
                solid_prob_img.append(result[i][0])
                solid_type_img.append(cls_dict[str(np.argmax(result[i][:])+1)])
            
            # write csv
            # img,index,img_width,img_height,prob,solid_prob,solid_type,points
            print('test loader image:',idx)
            lane_idx = 0
            for i in range(len(lines_per_img)):
                img.append(test_img_list['img'][idx])
                index.append(lane_idx)
                lane_idx = lane_idx + 1
                img_width.append(1280)
                img_height.append(720)
                lane_prob.append(lane_prob_per_img[i])
                solid_prob.append(solid_prob_img[i])
                solid_type.append(solid_type_img[i])
                points.append(lines_per_img[i])
            idx = idx + 1
        
        cont_list = {'img':img, 'index':index, 'img_width':img_width, 'img_height':img_height,
                    'prob':lane_prob,'solid_prob':solid_prob, 'solid_type':solid_type,'points':points}
        df = pd.DataFrame(cont_list)
        df.to_csv('result_test.csv',index=False)

class RBDataset(Dataset):
    def __init__(self,pf,transform=None):
        self.pf=pf
        self.transform=transform
        self.IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        self.IMAGENET_STD = np.array([0.229, 0.224, 0.225])
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.pf['img'])

    def __getitem__(self,idx):
        image_path = '/fs/scratch/ccserver_cc_cr_challenge/lane-detection/test/images/'+self.pf['img'][idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640,360)) #w,h  
        image = image / 255.
        image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        image = self.tran(image.astype(np.float32))
        #image=self.transform(Image.fromarray(image))
        return image, image_path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    cfg = Config('jst_b0.yaml')
    
    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    batch_size = cfg["batch_size"]

    # Model
    origin_model = cfg.get_model()
    origin_model = origin_model.cuda()
    model = JSTNET(origin_model, 20) # 2 class for each lane, 10 lane in total
    model = model.cuda()

    # Get data set
    # transform is useless
    test_trainsform=transforms.Compose([
            transforms.Resize([360,640]), # h,w
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    temp = pd.read_csv('/home/jns2szh/code/PolyLaneNet-master/test.csv')
    test_dataset = RBDataset(temp,test_trainsform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

    test(model, test_loader, cfg)
