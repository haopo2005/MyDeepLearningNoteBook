from main.model.squeezeDet import  SqueezeDet
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from main.model.visualization import  bbox_transform_single_box
from main.model.evaluation import filter_batch
import os
import sys
import time
import numpy as np
import cv2
import argparse
from main.config.create_config import load_dict


checkpoint_dir = './log/checkpoints/model.86-0.66.hdf5'
CUDA_VISIBLE_DEVICES = "0"
CONFIG = "squeeze.config"
def eval(img_file):
    #create config object
    config = load_dict(CONFIG)
    config.BATCH_SIZE = 1 #文件数量
    config.FINAL_THRESHOLD = 0.5 #conf阈值
    '''
    #open files with images and ground truths files with full path names
    with open(imgs_list) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    '''
    
    #hide the other gpus so tensorflow only uses this one
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


    #tf config and session
    cfg = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=cfg)
    K.set_session(sess)

    #instantiate model
    squeeze = SqueezeDet(config) 
    squeeze.model.load_weights(checkpoint_dir)

    img = cv2.imread(img_file).astype(np.float32, copy=False)
    orig_h, orig_w, _ = [float(v) for v in img.shape]
    # scale image
    draw_img = img
    img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    img = (img - np.mean(img))/ np.std(img)
    img = np.reshape(img, (1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    y_pred = squeeze.model.predict(img)
    #filter batch with nms
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, config)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, det_box in enumerate(all_filtered_boxes[0]):
        #transform into xmin, ymin, xmax, ymax
        det_box = bbox_transform_single_box(det_box)
        print(all_filtered_scores[0][j])
        #add rectangle and text
        cv2.rectangle(draw_img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (0,0,255), 1)
        cv2.putText(draw_img, 'head'+str(all_filtered_scores[0][j]), (det_box[0], det_box[1]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('restul.png',draw_img)

if __name__ == "__main__":
    #default values for some variables
    eval(sys.argv[1])