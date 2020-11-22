import sys
import os
import pandas as pd


def find_img(dir_path, img_list, lb_list, label):
    img_paths = os.listdir(dir_path)
    for img_path in img_paths:
        print(img_path)
        img_list.append(dir_path+img_path)
        lb_list.append(label)
    return img_list, lb_list

if __name__ == '__main__':
    img_list = []
    lb_list = []
    base_path = '/home/jns2szh/code/Basic_CNNs_TensorFlow2-master/dataset/valid/'
    img_list, lb_list = find_img(base_path+'0/', img_list, lb_list, 0)
    img_list, lb_list = find_img(base_path+'1/', img_list, lb_list, 1)
    img_list, lb_list = find_img(base_path+'2/', img_list, lb_list, 2)
    img_list, lb_list = find_img(base_path+'3/', img_list, lb_list, 3)
    
    cont_list = {'img_list':img_list, 'lb_list':lb_list}
    df = pd.DataFrame(cont_list, columns=['img_list','lb_list'])
    df.to_csv('valid.csv')
