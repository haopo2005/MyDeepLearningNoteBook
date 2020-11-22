import sys
import os
import argparse
import random
import time
import warnings
import math
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from config import Config
from public.imagenet import models
from public.imagenet.utils import DataPrefetcher, get_logger, AverageMeter, accuracy
from collections import OrderedDict


class JSTNET(nn.Module):
    def __init__(self, model_b3, model_12GF, model_32GF, nb_class=4):
    #def __init__(self, model_b3, model_12GF, model_32GF, model_vovnet, nb_class=4):
        super(JSTNET, self).__init__()
        self.model_b3 = model_b3
        self.model_12GF = model_12GF
        self.model_32GF = model_32GF
        #self.model_vovnet = model_vovnet
        
        self.model_b3.fc = nn.Identity()
        self.model_12GF.fc = nn.Identity()
        self.model_32GF.fc = nn.Identity()
        #self.model_vovnet.fc = nn.Identity()
        
        #self.dropout = nn.Dropout(0.5)
        #self.relu = nn.ReLU(inplace=True)
        #self.classifier = nn.Linear(8512, nb_class)
        self.classifier = nn.Linear(7488, nb_class)
        #self.classifier_2 = nn.Linear(200, nb_class)
        
    def forward(self, inputs):
        output_b3 = self.model_b3(inputs.clone())
        output_b3 = output_b3.view(output_b3.size(0), -1)
        output_12GF = self.model_12GF(inputs.clone())
        output_12GF = output_12GF.view(output_12GF.size(0), -1)
        #output_vovnet = self.model_vovnet(inputs.clone())
        #output_vovnet = output_vovnet.view(output_vovnet.size(0), -1)
        output_32GF = self.model_32GF(inputs.clone())
        output_32GF = output_32GF.view(output_32GF.size(0), -1)
        
        #new_input = torch.cat([output_b3,output_12GF,output_vovnet,output_32GF],dim=1)
        new_input = torch.cat([output_b3,output_12GF,output_32GF],dim=1)
        x = self.classifier(new_input)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.classifier_2(x)
        '''
        print('new_input:',new_input.shape)
        test_input = new_input.view(new_input.shape[0],-1)
        print('test_inpu:',test_input.shape)
        (b,in_f) = test_input.shape
        print(b, in_f)
        
        self.fc1 = nn.Linear(in_f, 100) 
        x = self.fc1(new_input)
        x = self.relu(x)
        x = self.dropout(x)
        test_input = x.view(x.size[0],-1)
        (b,in_f) = test_input.shape
        self.fc2 = nn.Linear(in_f,4)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
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

def validate(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    result = 0
    with torch.no_grad():
        end = time.time()
        labels_len = 0
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # compute softmax loss
            pred_score = outputs.cuda().data.cpu().numpy()
            for i in range(len(labels)):
                for j in range(4):
                    if labels[i] == j and pred_score[i][j]>0:
                        result += math.log(pred_score[i][j])
            labels_len += len(labels)
        result = -result/labels_len
        logger.info(f'rb loss: {result}')
    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, top5.avg, throughput,result


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()
    
    
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(Config.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    logger.info('finish loading data')
    
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
    '''
    model_vovnet = models.__dict__['VoVNet99_se'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
    '''
    model_32GF = models.__dict__['RegNetY_32GF'](**{
                    "pretrained": False,
                    "num_classes": 4,
                    })
    
    for name, param in model_b3.named_parameters():
        param.requires_grad = False
        logger.info(f"{name},{param.requires_grad}")
    
    for name, param in model_12GF.named_parameters():
        param.requires_grad = False
        logger.info(f"{name},{param.requires_grad}")
    '''
    for name, param in model_vovnet.named_parameters():
        param.requires_grad = False
        logger.info(f"{name},{param.requires_grad}")
    '''
    for name, param in model_32GF.named_parameters():
        param.requires_grad = False
        logger.info(f"{name},{param.requires_grad}")
    
    # merge model
    logger.info(f"creating ensemble model")
    #model = JSTNET(model_b3, model_12GF, model_32GF, model_vovnet)
    model = JSTNET(model_b3, model_12GF, model_32GF)
    model = model.cuda()
    model_b3 = model_b3.cuda()
    model_12GF = model_12GF.cuda()
    model_32GF = model_32GF.cuda()
    #model_vovnet = model_vovnet.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
        math.cos((epoch - args.warm_up_epochs) /
                 (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_up_with_cosine_lr)
        
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    
    model = nn.DataParallel(model)
    
     #load model
    my_path = '/home/jns2szh/code/pytorch-ImageNet-CIFAR-COCO-VOC-training-master/imagenet_experiments/'
    logger.info(f"start load model")
    checkpoint_b3 = torch.load(my_path+'efficientnet_imagenet_DataParallel_train_example/checkpoints_b3/latest.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint_b3['model_state_dict'].items():
        name = k[7:] # remove module.
        if name != 'fc.weight' and name != 'fc.bias':
            new_state_dict[name] = v
    model_b3.load_state_dict(new_state_dict)
    logger.info(f"load b3 model finished")
    
    checkpoint_12GF = torch.load(my_path+'regnet_imagenet_Dataparallel_train_example/regnet_12/latest.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint_12GF['model_state_dict'].items():
        name = k[7:] # remove module.
        if name != 'fc.weight' and name != 'fc.bias':
            new_state_dict[name] = v
    model_12GF.load_state_dict(new_state_dict)
    logger.info(f"load 12GF model finished")
    
    '''
    checkpoint_vovnet = torch.load(my_path+'vovnet_Dataparallel_train_example/checkpoints/latest.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint_vovnet['model_state_dict'].items():
        name = k[7:] # remove module.
        if name != 'fc.weight' and name != 'fc.bias':
            new_state_dict[name] = v
    model_vovnet.load_state_dict(new_state_dict)
    logger.info(f"load vovnet model finished")
    '''
    checkpoint_32GF = torch.load(my_path+'regnet_imagenet_Dataparallel_train_example/checkpoints/latest.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint_32GF['model_state_dict'].items():
        name = k[7:] # remove module.
        if name != 'fc.weight' and name != 'fc.bias':
            new_state_dict[name] = v
    model_32GF.load_state_dict(new_state_dict)
    logger.info(f"load 32GF model finished")
    
    # resume training
    start_epoch=0
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%")
    
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    min_rb_loss = 1000
    
    for epoch in range(start_epoch, args.epochs + 1):
        #print(epoch, logger,args)
        '''
        acc1, losses = train(train_loader, model, criterion, optimizer, 
                                epoch, 
                                logger)
        '''
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        model.train()

        iters = len(train_loader.dataset) // args.batch_size
        prefetcher = DataPrefetcher(train_loader)
        inputs, labels = prefetcher.next()
        iter_index = 1

        while inputs is not None:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / 1

            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if iter_index % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            inputs, labels = prefetcher.next()

            if iter_index % args.print_interval == 0:
                logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
                )

            iter_index += 1
        
        scheduler.step()
        '''
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, losses: {losses:.2f}"
        )
        '''

        acc1, acc5, throughput, rb_loss = validate(val_loader, model)
        logger.info(
            f"val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        
        if rb_loss < min_rb_loss:
            min_rb_loss = rb_loss
            logger.info("save model")
            torch.save(
                {
                'epoch': epoch,
                'acc1': acc1,
                'loss': losses,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoints, 'latest.pth'))
        if epoch == args.epochs:
            torch.save(
                {
                'epoch': epoch,
                'acc1': acc1,
                'loss': losses,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                },
                os.path.join(
                    args.checkpoints,
                    "{}-epoch{}-acc{}.pth".format('JSTNET', epoch, acc1)))

    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7"
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)
