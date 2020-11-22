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

# extract points for single lane and lane conf
def pred2lanes(pred, image_path, img_h, img_w):
    lanes = []
    # score + upper + lower + 6 coeffs = 9
    lane = pred[1:]  # remove conf
    lower, upper = lane[0], lane[1] #max_y, min_y
    lane = lane[2:]  # remove upper, lower positions --> polynomial coefficients
    
    # generate points from the polynomial
    #print(lower, upper, upper_shared)
    ys = np.linspace(upper, lower, num=15)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(float)
    points[:, 0] = (np.polyval(lane, ys) * img_w).astype(float)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)].tolist()
    
    #filter bad/repeat points
    for i in range(len(points)-1):
        temp = points[i]
        temp2 = points[i+1]
        if temp[0] == temp2[0] and temp[1] == temp2[1]:
            print('find repeated points, image path:',image_path)
            points = []
            return points, pred[0]
    return points, pred[0]

def generate_lines_per_img(predictions, image_path):
    lines_per_batch = []
    conf_batch = []
    upper_fist_lane = predictions[0][2]
    for idx in range(len(predictions)):
        points, conf = pred2lanes(predictions[idx], image_path, 720, 1280) #points for each image
        lines_per_batch.append(points)
        conf_batch.append(conf)
    return lines_per_batch, conf_batch



cls_dict = {'1': 'solid', '2': 'dashed', '3': 'unknown'}

def test(model, test_loader, cfg):
    logging.info("Starting testing.")
    epoch = 783 #trained epoches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join('/home/jns2szh/code/PolyLaneNet-master/experiments/bosch_5_order', "models", "model_{:03d}.pt".format(epoch)))['model'])
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
            lines_per_img, lane_prob_per_img = generate_lines_per_img(outputs.cuda().data.cpu().numpy(), image_path)

            solid_prob_img = []
            solid_type_img = []
            result = lane_cls.cuda().data.cpu().numpy() # 1*10*2
            for i in range(len(result)):
                solid_prob_img.append(result[i][0])
                solid_type_img.append(cls_dict[str(np.argmax(result[i][:])+1)])
            
            # write csv
            # img,index,img_width,img_height,prob,solid_prob,solid_type,points
            print('test loader image:',image_path)
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
        df.to_csv('result.csv',index=False)

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
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
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
    model = cfg.get_model().to(device)

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
                                              num_workers=8)

    test(model, test_loader, cfg)
