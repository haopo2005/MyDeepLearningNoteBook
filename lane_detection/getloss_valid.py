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

    def loss(self,
             outputs,
             target,
             conf_weight=1,
             lower_weight=1,
             upper_weight=1,
             cls_weight=1,
             poly_weight=300,
             threshold=15 / 720.):
        pred, extra_outputs = outputs
        bce = nn.BCELoss()
        mse = nn.MSELoss()
        s = nn.Sigmoid()
        threshold = nn.Threshold(threshold**2, 0.)
        pred = pred.reshape(-1, target.shape[1], 1 + 2 + 4) # target: 3 dimesion, target.shape[1]->max_lanes
        target_categories, pred_confs = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
        target_uppers, pred_uppers = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
        target_points, pred_polys = target[:, :, 3:].reshape((-1, target.shape[2] - 3)), pred[:, :, 3:].reshape(-1, 4) #4 poly coeffs
        target_lowers, pred_lowers = target[:, :, 1], pred[:, :, 1] #batch, lanes, lower_idx

        target_lowers = target_lowers.reshape((-1, 1))
        pred_lowers = pred_lowers.reshape((-1, 1))

        target_confs = (target_categories > 0).float() # if class id larger than 0
        valid_lanes_idx = target_confs == 1 # select the valid lane, because pred has fake lane
        valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)
        lower_loss = mse(target_lowers[valid_lanes_idx], pred_lowers[valid_lanes_idx])
        upper_loss = mse(target_uppers[valid_lanes_idx], pred_uppers[valid_lanes_idx])

        # classification loss
        if self.pred_category and self.extra_outputs > 0:
            ce = nn.CrossEntropyLoss()
            pred_categories = extra_outputs.reshape(target.shape[0] * target.shape[1], -1)
            target_categories = target_categories.reshape(pred_categories.shape[:-1]).long()
            pred_categories = pred_categories[target_categories > 0]
            target_categories = target_categories[target_categories > 0]
            cls_loss = ce(pred_categories, target_categories - 1)
        else:
            cls_loss = 0

        # poly loss calc
        target_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2] #divide and floor
        ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:].t()
        valid_xs = target_xs >= 0
        pred_polys = pred_polys[valid_lanes_idx_flat]
        # polys order is fixed here
        pred_xs = pred_polys[:, 0] * ys**3 + pred_polys[:, 1] * ys**2 + pred_polys[:, 2] * ys + pred_polys[:, 3]
        pred_xs.t_()
        # sqrt(total sum of valid_xs divided by col sum of valid_xs)
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        pred_xs = (pred_xs.t_() *
                   weights).t()  # without this, lanes with more points would have more weight on the cost function
        target_xs = (target_xs.t_() * weights).t()
        poly_loss = mse(pred_xs[valid_xs], target_xs[valid_xs]) / valid_lanes_idx.sum()
        poly_loss = threshold(
            (pred_xs[valid_xs] - target_xs[valid_xs])**2).sum() / (valid_lanes_idx.sum() * valid_xs.sum())

        # applying weights to partial losses
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight
        conf_loss = bce(pred_confs, target_confs) * conf_weight

        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss

        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss
        }

# extract points for single lane and lane conf
def pred2lanes(pred, upper_shared, img_h, img_w):
    lanes = []
    # score + upper + lower + 4 coeffs = 7
    lane = pred[1:]  # remove conf
    lower, upper = lane[0], lane[1] #max_y, min_y
    lane = lane[2:]  # remove upper, lower positions --> polynomial coefficients
    
    # generate points from the polynomial
    #print(lower, upper, upper_shared)
    ys = np.linspace(upper, lower, num=15)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(int)
    points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)].tolist()
    
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

def get_result(model, test_loader, cfg):
    logging.info("Starting testing.")
    epoch = 54 #trained epoches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join('/home/jns2szh/code/PolyLaneNet-master/experiments/bosch_3_class', "models", "model_{:03d}.pt".format(epoch)))['model'])
    model.eval()


    criterion_parameters = cfg.get_loss_parameters()
    test_parameters = cfg.get_test_parameters()
    criterion = model.loss
    loss = 0
    total_iters = 0

    test_parameters = cfg.get_test_parameters()
    with torch.no_grad():
        test_img_list = pd.read_csv('/home/jns2szh/code/PolyLaneNet-master/train_filtered.csv')
        img = []
        index = []
        img_width = []
        img_height = []
        lane_prob = []
        solid_prob = []
        points = []
        solid_type = []
        valid_loss = []

        #tips: only one image each time
        idx = 0
        for _, (images, labels, img_idxs) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_i, loss_dict_i = criterion(outputs, labels, **criterion_parameters)

            outputs, extra_outputs = model.decode(outputs, 1, **test_parameters) #assume batch size is 1
            lane_cls = F.softmax(extra_outputs, dim=1)
            lines_per_img, lane_prob_per_img = generate_lines_per_img(outputs.cuda().data.cpu().numpy())

            solid_prob_img = []
            solid_type_img = []
            result = lane_cls.cuda().data.cpu().numpy() # 1*10*2
            for i in range(len(result)):
                solid_prob_img.append(result[i][0])
                if result[i][0] > result[i][1]:
                    solid_type_img.append('solid')
                else:
                    solid_type_img.append('dashed')
            
            # write csv
            # img,index,img_width,img_height,prob,solid_prob,solid_type,points
            print('test loader image:',str(idx))
            lane_idx = 0
            for i in range(len(lines_per_img)):
                valid_loss.append(loss_i.item())
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
        cont_list = {'valid_loss':valid_loss,'img':img, 'index':index, 'img_width':img_width, 'img_height':img_height,
                    'prob':lane_prob,'solid_prob':solid_prob, 'solid_type':solid_type,'points':points}
        df = pd.DataFrame(cont_list)
        df.to_csv('result_train.csv',index=False)

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
    model = JSTNET(origin_model, 30) # 3 class for each lane, 10 lane in total
    model = model.cuda()

    # Get data set
    val_dataset = cfg.get_dataset("train")
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=8)

    get_result(model, val_loader, cfg)
