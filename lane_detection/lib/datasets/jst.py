import os
import json
import random

import numpy as np
from tabulate import tabulate
import pandas as pd
from utils.lane import LaneEval
from utils.metric import eval_json

SPLIT_FILES = {
    'train': ['train.csv'],
    'val': ['valid.csv']
}

class JSTIMAGE:
    def __init__(self):
        self.solid_type = []
        self.points_list = []
        self.img_path = ''
        self.points_size = 0
    
    def clear(self):
        self.solid_type = []
        self.points_list = []
        self.img_path = ''
        self.points_size = 0

class JST(object):
    def __init__(self, split='train', max_lanes=None, root=None, metric='default'):
        self.split = split
        self.root = root
        self.metric = metric

        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        self.anno_files = [os.path.join('/home/jns2szh/code/PolyLaneNet-master', path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 1280, 720
        self.max_points = 0
        # Force max_lanes, used when evaluating testing with models trained on other datasets
        self.max_lanes = 10
        self.load_annotations()

    def get_img_heigth(self, path):
        return 720

    def get_img_width(self, path):
        return 1280

    def get_metrics(self, lanes, idx):
        # Placeholders
        return [1] * len(lanes), [1] * len(lanes), None

    def load_annotations(self):
        self.annotations = []
        for anno_file in self.anno_files: #load train or valid, should be only one file here
            data = pd.read_csv(anno_file, index_col=False)
            idx = 0
            prev_img_name = ''
            jst_img_list = []
            test_img = None
            
            for i in range(len(data['img'])):
                tmp_img_name = data['img'][i]
                tmp_idx = data['index'][i]
                tmp_points = json.loads(data['points'][i])
                tmp_type = data['solid_type'][i]
                if tmp_type == 'solid':
                    tmp_type = 1
                elif tmp_type == 'dashed':
                    tmp_type = 2
                else:
                    tmp_type = 3 
                
                if idx == 0:
                    prev_img_name = tmp_img_name
                    test_img = JSTIMAGE()
                    test_img.img_path = tmp_img_name + '.bmp'
                    test_img.points_list.append(tmp_points)
                    test_img.solid_type.append(tmp_type)
                    test_img.points_size = test_img.points_size + len(tmp_points)
                    idx = idx + 1
                else:
                    if tmp_img_name == prev_img_name and idx == tmp_idx:
                        test_img.points_list.append(tmp_points)
                        test_img.solid_type.append(tmp_type)
                        test_img.points_size = test_img.points_size + len(tmp_points)
                        idx = idx + 1
                    else:
                        jst_img_list.append(test_img)
                        #self.max_lanes = max(self.max_lanes, idx)
                        idx = 0
                        prev_img_name = tmp_img_name
                        test_img = JSTIMAGE()
                        test_img.img_path = tmp_img_name + '.bmp'
                        test_img.points_list.append(tmp_points)
                        test_img.solid_type.append(tmp_type)
                        test_img.points_size = test_img.points_size + len(tmp_points)
                        idx = idx + 1
            jst_img_list.append(test_img) # in case of missing the last one
   
            for j in range(len(jst_img_list)):
                self.annotations.append({'lanes': jst_img_list[j].points_list, 'path': '/fs/scratch/ccserver_cc_cr_challenge/lane-detection/train/images/' + jst_img_list[j].img_path, 
                                        'categories': jst_img_list[j].solid_type})
                #print('points size per image:',jst_img_list[j].points_size)
                self.max_points = max(self.max_points, jst_img_list[j].points_size)

#        if self.split == 'train':
#            random.shuffle(self.annotations)
        print('total annos,', len(self.annotations))
        print('max points:', self.max_points)
        print('max lanes:', self.max_lanes)

    #useless
    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        pred_filename = 'jst_predictions_{}.csv'.format(label)
        self.save_tusimple_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        elif self.metric == 'ours':
            result = json.loads(eval_json(pred_filename, self.anno_files[0], json_type='tusimple'))
        table = {}
        for metric in result:
            table[metric['name']] = [metric['value']]
        table = tabulate(table, headers='keys')

        if not only_metrics:
            filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)

        return table, result

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
