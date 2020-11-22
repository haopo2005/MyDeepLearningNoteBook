import sys
import os
import argparse
import math
import random
import time
import warnings
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim
#from thop import profile
#from thop import clever_format
from torch.utils.data import DataLoader,Dataset
from config import Config
from public.imagenet import models
import cv2
from public.imagenet.utils import DataPrefetcher, get_logger, AverageMeter, accuracy
from PIL import Image
from torch.nn import functional as F

class JSTNET(nn.Module):
    def __init__(self, model_b3, model_12GF, model_32GF, model_vovnet, nb_class=4):
        super(JSTNET, self).__init__()
        self.model_b3 = model_b3
        self.model_12GF = model_12GF
        self.model_32GF = model_32GF
        self.model_vovnet = model_vovnet
        
        self.model_b3.fc = nn.Identity()
        self.model_12GF.fc = nn.Identity()
        self.model_32GF.fc = nn.Identity()
        self.model_vovnet.fc = nn.Identity()
        
        #self.dropout = nn.Dropout(0.5)
        #self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(8512, nb_class)
        #self.classifier_2 = nn.Linear(200, nb_class)
        
    def forward(self, inputs):
        output_b3 = self.model_b3(inputs.clone())
        output_b3 = output_b3.view(output_b3.size(0), -1)
        output_12GF = self.model_12GF(inputs.clone())
        output_12GF = output_12GF.view(output_12GF.size(0), -1)
        output_vovnet = self.model_vovnet(inputs.clone())
        output_vovnet = output_vovnet.view(output_vovnet.size(0), -1)
        output_32GF = self.model_32GF(inputs.clone())
        output_32GF = output_32GF.view(output_32GF.size(0), -1)
        
        new_input = torch.cat([output_b3,output_12GF,output_vovnet,output_32GF],dim=1)
        x = self.classifier(F.relu(new_input))
        '''
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier_2(x)
        '''
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--warm_up_epochs',
                        type=int,
                        default=Config.warm_up_epochs,
                        help='num of warm up epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')

    return parser.parse_args()

def test(val_loader, model, args):
    # switch to evaluate mode
    model.eval()
    
    dry = []
    wet = []
    snowy = []
    na = []
    count = 0
    #print(val_loader.size())
    for inputs in val_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        print(outputs.size(),inputs.size())
        result = outputs.cuda().data.cpu().numpy()
        for i in range(len(inputs)):
            dry.append(result[i][0])
            wet.append(result[i][1])
            snowy.append(result[i][2])
            na.append(result[i][3])
            print(count)
            count += 1
    cont_list = {'dry':dry, 'wet':wet, 'snowy':snowy, 'na':na}
    df = pd.DataFrame(cont_list, columns=['dry','wet','snowy','na'])
    df.to_csv('result.csv')


class RBDataset(Dataset):
    def __init__(self,pf,transform=None):
        self.pf=pf
        self.transform=transform

    def __len__(self):
        return len(self.pf['img_path'])

    def __getitem__(self,idx):
        image_path='/home/jns2szh/code/Basic_CNNs_TensorFlow2-master/'+self.pf['img_path'][idx]
        image=cv2.imread(image_path)
        image=self.transform(Image.fromarray(image))
        return image


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    print('used gpu:',gpus)
    
    cudnn.benchmark = True
    cudnn.enabled = True
    
    test_trainsform=transforms.Compose([
            transforms.Resize(int(384)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    test_data = RBDataset(pd.read_csv('/home/jns2szh/code/Basic_CNNs_TensorFlow2-master/inference.csv'),test_trainsform)
    # dataset and dataloader
    logger.info('start loading data')
    test_loader = DataLoader(test_data,
                            batch_size=8,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    print('finish loading data')

    if not os.path.isfile(args.evaluate):
        raise Exception(
            f"{args.resume} is not a file, please check it again")
    print('start only evaluating')
    print("start resuming model from "+args.evaluate)
    checkpoint = torch.load(args.evaluate,
                            map_location=torch.device('cpu'))
    
    # dataset and dataloader
    logger.info('start loading data')
    model_b3 = models.__dict__['efficientnet_b3'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
        
    model_12GF = models.__dict__['RegNetY_12GF'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
        
    model_vovnet = models.__dict__['VoVNet99_se'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
        
    model_32GF = models.__dict__['RegNetY_32GF'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
    
    model = JSTNET(model_b3, model_12GF, model_32GF, model_vovnet)
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(test_loader, model, args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)
