import tensorflow as tf
from configuration import save_model_dir, test_image_dir
from prepare_data import load_and_preprocess_image
from train import get_model
import sys
import pandas as pd
import numpy as np
import os

def get_single_picture_prediction(model, picture_dir):
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_dir), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    #print(np.shape(prediction.numpy()))
    return prediction.numpy()

if __name__ == '__main__':

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # load the model
    model = get_model()
    model.load_weights('saved_model_cbam/epoch-40')
    # model = tf.saved_model.load(save_model_dir)

    csv_path = sys.argv[1]
    pf = pd.read_csv(csv_path)
    img_path = pf['img_list']
    
    pred_score = []
    label_list = []
    print(len(img_path))
    for i in range(len(img_path)):
        pred = get_single_picture_prediction(model, img_path[i])
        pred_score.append(pred[0])
        label_list.append(pf['lb_list'][i])
        #print(pred[0])
    #print('fucked')
    bad_img_id = []
    for i in range(len(label_list)):
        lb = label_list[i]
        pred_lb = np.argmax(pred_score[i])
        if lb != pred_lb:
            print(lb, pred_lb)
            cmd = 'cp ' + img_path[i] + ' /home/jns2szh/code/Basic_CNNs_TensorFlow2-master/dataset/bad_data/'+str(lb)
            print(cmd)
            os.system(cmd)
