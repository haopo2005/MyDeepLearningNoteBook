import os,sys
from skimage import io
import json
import pandas as pd

if __name__ == '__main__':
    csv_path = 'origin_test_filter.csv'
    new_path = 'new_test.csv'
    filter_list = []
    
    data = pd.read_csv(csv_path, index_col=False)
    df2 = data.copy()
    
    for i in range(len(data['img'])):
        img_path = '/fs/scratch/ccserver_cc_cr_challenge/lane-detection/test/images/' + data['img'][i]
        #print(img_path)
        try:
            img = io.imread(img_path)
        except:
            print('image broken:',img_path)
            filter_list.append(img_path)
            df2.drop(df2.index[i])
            
    
    cont_list = {'broken_list':filter_list}
    df = pd.DataFrame(cont_list, columns=['broken_list'])
    df.to_csv('broken_test.csv')

    df2.to_csv(new_path, header=1)
