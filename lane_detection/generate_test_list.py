import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    base_path = '/fs/scratch/ccserver_cc_cr_challenge/lane-detection/test/images/'
    img_list = os.listdir(base_path)
    cont_list = {'img':img_list}
    df = pd.DataFrame(cont_list, columns=['img'])
    df.to_csv('test.csv',index=False)