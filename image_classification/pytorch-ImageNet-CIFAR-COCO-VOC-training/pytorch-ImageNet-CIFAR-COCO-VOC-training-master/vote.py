import os
import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':
    #b4_csv = 'b7.csv'
    #shuffle_csv = 'vovnet_99_250.csv'
    #regnet_csv = 'regnet_32_250.csv'
    #b4_csv = 'b7_reg32_vov99.csv'
    #shuffle_csv = 'result_60_epoch.csv'
    #regnet_csv = 'b3_imagenet_250.csv'
    
    '''
    b4_csv = 'merged_xinyun.csv' #resnet50_12 + 6GF + vovnet_99
    #shuffle_csv = 'reg_12_224.csv' 
    #regnet_csv = 'b7.csv'
    shuffle_csv = 'b3_imagenet_250.csv' #b3_imagenet_250, result_60_epoch,regnet_32_250.csv
    regnet_csv = 'result_60_epoch.csv'
    '''
    #b4_csv = 'merged_b7_99_6GF.csv'
    #shuffle_csv = 'resnet-50.csv'
    #regnet_csv = 'regnet_32_250.csv'
    
    
    b4_csv = 'yafei.csv'
    shuffle_csv = 'ResNet50_12.csv'
    regnet_csv = '99_lr.csv'
    
    pf_b4 = pd.read_csv(b4_csv)
    pf_shuffle = pd.read_csv(shuffle_csv)
    pf_regnet = pd.read_csv(regnet_csv)
    
    dry = []
    wet = []
    snowy = []
    na = []
    
    for i in range(len(pf_b4['id'])):
        temp_shuffle = np.array([pf_shuffle['dry'][i], pf_shuffle['wet'][i],pf_shuffle['snowy'][i], pf_shuffle['na'][i]])
        temp_regnet = np.array([pf_regnet['dry'][i], pf_regnet['wet'][i],pf_regnet['snowy'][i], pf_regnet['na'][i]])
        temp_b4 = np.array([pf_b4['dry'][i], pf_b4['wet'][i],pf_b4['snowy'][i], pf_b4['na'][i]])
        
        idx_shuffle = np.argmax(temp_shuffle)
        idx_regnet = np.argmax(temp_regnet)
        
        if idx_regnet == idx_shuffle:
            temp_b4[idx_regnet] = 1
        
        dry.append(temp_b4[0])
        wet.append(temp_b4[1])
        snowy.append(temp_b4[2])
        na.append(temp_b4[3])
    cont_list = {'dry':dry, 'wet':wet, 'snowy':snowy, 'na':na}
    df = pd.DataFrame(cont_list, columns=['dry','wet','snowy','na'])
    df.to_csv('merged.csv')