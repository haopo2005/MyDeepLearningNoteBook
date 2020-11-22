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
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # load the model
    model = get_model()
    model.load_weights("saved_model_shufflev2/model")
    csv_path = sys.argv[1]
    pf = pd.read_csv(csv_path)
    img_path = pf['img_path']
    dry = []
    wet = []
    snowy = []
    na = []
    for i in range(len(img_path)):
        img_src = '/home/jns2szh/code/Basic_CNNs_TensorFlow2-master/'+img_path[i]
        prediction_score = get_single_picture_prediction(model, img_src)
        print(prediction_score)
        dry.append(prediction_score[0][0])
        wet.append(prediction_score[0][1])
        snowy.append(prediction_score[0][2])
        na.append(prediction_score[0][3])
    cont_list = {'dry':dry, 'wet':wet, 'snowy':snowy, 'na':na}
    df = pd.DataFrame(cont_list, columns=['dry','wet','snowy','na'])
    df.to_csv('result.csv')
