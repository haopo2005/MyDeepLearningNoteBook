import os
import sys
import random
import shutil
import logging
import argparse
import subprocess
from time import time

import numpy as np
import torch

from test import test
from lib.config import Config
from utils.evaluator import Evaluator
import torch.nn as nn
from collections import OrderedDict
import pandas as pd

class JSTNET(nn.Module):
    def __init__(self, model_elas_cls, extra_outputs):
        super(JSTNET, self).__init__()
        self.model = model_elas_cls
        for name, param in self.model.named_parameters():
            print(name, param.size())
        # self.model.model._fc.regular_outputs_layer.in_features --> 1280
        self.model.model._fc.regular_outputs_layer = nn.Linear( self.model.model._fc.regular_outputs_layer.in_features, 70)
        self.model.model._fc.extra_outputs_layer = nn.Linear( self.model.model._fc.regular_outputs_layer.in_features, 30)
        self.share_top_y = True
        self.curriculum_steps = [0, 0, 0, 0]
        self.pred_category = True
        self.extra_outputs = 30
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        output, extra_outputs = self.model(inputs)
        return output, extra_outputs
    
    def decode(self, all_outputs, batch_size, conf_threshold=0.5):
        outputs, extra_outputs = all_outputs
        if extra_outputs is not None:
            extra_outputs = extra_outputs.reshape(batch_size, 10, -1)
            #extra_outputs = extra_outputs.argmax(dim=2)
        outputs = outputs.reshape(len(outputs), -1, 7)  # score + upper + lower + 4 coeffs = 7
        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])
        result_outputs = outputs[outputs[:, :, 0] > conf_threshold]
        result_extra_outputs = extra_outputs[outputs[:, :, 0] > conf_threshold]

        if False and self.share_top_y:
            outputs[:, :, 0] = outputs[:, 0, 0].expand(outputs.shape[0], outputs.shape[1])

        return result_outputs, result_extra_outputs

    def loss(self,
             outputs,
             target,
             conf_weight=1,
             lower_weight=1,
             upper_weight=1,
             cls_weight=1,
             poly_weight=300,
             threshold=15 / 720.):
        pred, extra_outputs = outputs
        bce = nn.BCELoss()
        mse = nn.MSELoss()
        s = nn.Sigmoid()
        threshold = nn.Threshold(threshold**2, 0.)
        pred = pred.reshape(-1, target.shape[1], 1 + 2 + 4) # target: 3 dimesion, target.shape[1]->max_lanes
        target_categories, pred_confs = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
        target_uppers, pred_uppers = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
        target_points, pred_polys = target[:, :, 3:].reshape((-1, target.shape[2] - 3)), pred[:, :, 3:].reshape(-1, 4) #4 poly coeffs
        target_lowers, pred_lowers = target[:, :, 1], pred[:, :, 1]

        '''
        if self.share_top_y:
            # inexistent lanes have -1e-5 as lower
            # i'm just setting it to a high value here so that the .min below works fine
            target_lowers[target_lowers < 0] = 1
            target_lowers[...] = target_lowers.min(dim=1, keepdim=True)[0]
            pred_lowers[...] = pred_lowers[:, 0].reshape(-1, 1).expand(pred.shape[0], pred.shape[1])
        '''
        
        target_lowers = target_lowers.reshape((-1, 1))
        pred_lowers = pred_lowers.reshape((-1, 1))

        target_confs = (target_categories > 0).float() # if class id larger than 0
        valid_lanes_idx = target_confs == 1 # select the valid lane, because pred has fake lane
        valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)
        lower_loss = mse(target_lowers[valid_lanes_idx], pred_lowers[valid_lanes_idx])
        upper_loss = mse(target_uppers[valid_lanes_idx], pred_uppers[valid_lanes_idx])

        # classification loss
        if self.pred_category and self.extra_outputs > 0:
            ce = nn.CrossEntropyLoss()
            pred_categories = extra_outputs.reshape(target.shape[0] * target.shape[1], -1)
            target_categories = target_categories.reshape(pred_categories.shape[:-1]).long()
            pred_categories = pred_categories[target_categories > 0]
            target_categories = target_categories[target_categories > 0]
            cls_loss = ce(pred_categories, target_categories - 1)
        else:
            cls_loss = 0

        # poly loss calc
        target_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2] #divide and floor
        ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:].t()
        valid_xs = target_xs >= 0
        pred_polys = pred_polys[valid_lanes_idx_flat]
        # polys order is fixed here
        pred_xs = pred_polys[:, 0] * ys**3 + pred_polys[:, 1] * ys**2 + pred_polys[:, 2] * ys + pred_polys[:, 3]
        pred_xs.t_()
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        pred_xs = (pred_xs.t_() *
                   weights).t()  # without this, lanes with more points would have more weight on the cost function
        target_xs = (target_xs.t_() * weights).t()
        poly_loss = mse(pred_xs[valid_xs], target_xs[valid_xs]) / valid_lanes_idx.sum()
        poly_loss = threshold(
            (pred_xs[valid_xs] - target_xs[valid_xs])**2).sum() / (valid_lanes_idx.sum() * valid_xs.sum())

        # applying weights to partial losses
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight
        conf_loss = bce(pred_confs, target_confs) * conf_weight

        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss

        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss
        }

def train(model, train_loader, exp_dir, cfg, val_loader, train_state=None):
    # Get initial train state
    optimizer = cfg.get_optimizer(model.parameters())
    scheduler = cfg.get_lr_scheduler(optimizer)
    starting_epoch = 1

    if train_state is not None:
        model.load_state_dict(train_state['model'])
        #optimizer.load_state_dict(train_state['optimizer'])#
        #scheduler.load_state_dict(train_state['lr_scheduler'])#
        starting_epoch = train_state['epoch'] + 1
        #scheduler.step(starting_epoch)#

    # Train the model
    criterion_parameters = cfg.get_loss_parameters()
    criterion = model.loss
    total_step = len(train_loader)
    ITER_LOG_INTERVAL = cfg['iter_log_interval']
    ITER_TIME_WINDOW = cfg['iter_time_window']
    MODEL_SAVE_INTERVAL = cfg['model_save_interval']
    t0 = time()
    total_iter = 0
    iter_times = []
    logging.info("Starting training.")
    for epoch in range(starting_epoch, num_epochs + 1):
        epoch_t0 = time()
        logging.info("Beginning epoch {}".format(epoch))
        accum_loss = 0
        for i, (images, labels, img_idxs) in enumerate(train_loader):
            total_iter += 1
            iter_t0 = time()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            accum_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_times.append(time() - iter_t0)
            if len(iter_times) > 100:
                iter_times = iter_times[-ITER_TIME_WINDOW:]
            if (i + 1) % ITER_LOG_INTERVAL == 0:
                loss_str = ', '.join(
                    ['{}: {:.4f}'.format(loss_name, loss_dict_i[loss_name]) for loss_name in loss_dict_i])
                logging.info("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({}), s/iter: {:.4f}, lr: {:.1e}".format(
                    epoch,
                    num_epochs,
                    i + 1,
                    total_step,
                    accum_loss / (i + 1),
                    loss_str,
                    np.mean(iter_times),
                    optimizer.param_groups[0]["lr"],
                ))
        logging.info("Epoch time: {:.4f}".format(time() - epoch_t0))
        if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == num_epochs:
            model_path = os.path.join(exp_dir, "models", "model_{:03d}.pt".format(epoch))
            save_train_state(model_path, model, optimizer, scheduler, epoch)
        if val_loader is not None:
            evaluator = Evaluator(val_loader.dataset, exp_root)
            evaluator, val_loss = test(
                model,
                val_loader,
                evaluator,
                None,
                cfg,
                view=False,
                epoch=-1,
                verbose=False,
            )
            logging.info("Epoch [{}/{}], Val loss: {:.4f}".format(epoch, num_epochs, val_loss))
            model.train()
        scheduler.step()
    logging.info("Training time: {:.4f}".format(time() - t0))

    return model


def save_train_state(path, model, optimizer, lr_scheduler, epoch):
    train_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }

    torch.save(train_state, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PolyLaneNet")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--validate", action="store_true", help="Validate model during training")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")

    return parser.parse_args()

def setup_exp_dir(exps_dir, exp_name, cfg_path):
    dirs = ["models"]
    exp_root = os.path.join(exps_dir, exp_name)

    for dirname in dirs:
        os.makedirs(os.path.join(exp_root, dirname), exist_ok=True)

    shutil.copyfile(cfg_path, os.path.join(exp_root, 'config.yaml'))

    return exp_root


def get_exp_train_state(exp_root):
    models_dir = os.path.join(exp_root, "models")
    models = os.listdir(models_dir)
    last_epoch, last_modelname = sorted(
        [(int(name.split("_")[1].split(".")[0]), name) for name in models],
        key=lambda x: x[0],
    )[-1]
    train_state = torch.load(os.path.join(models_dir, last_modelname))

    return train_state


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up experiment
    if not args.resume:
        exp_root = setup_exp_dir(cfg['exps_dir'], args.exp_name, args.cfg)
    else:
        exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    #logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Get data sets
    train_dataset = cfg.get_dataset("train")

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    # Model
    origin_model = cfg.get_model()
    origin_model = origin_model.cuda()

    
    for k, param in origin_model.named_parameters():
        print(k)
    '''
        if k!='model._fc.regular_outputs_layer.weight' and \
                        k!='model._fc.regular_outputs_layer.bias' and k!='model._fc.extra_outputs_layer.weight' and \
                        k!='model._fc.extra_outputs_layer.bias':
            param.requires_grad = False
    '''
    
    model = JSTNET(origin_model, 20) # 2 class for each lane, 10 lane in total
    model = model.cuda()
    
    model_dict = origin_model.state_dict()
    pretrained_dict = torch.load(os.path.join('/home/jns2szh/code/PolyLaneNet-master/experiments/bosch_2_class/models', "model_{:03d}.pt".format(263)))['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k!='model._fc.regular_outputs_layer.weight' and 
                        k!='model._fc.regular_outputs_layer.bias' and k!='model._fc.extra_outputs_layer.weight' and 
                        k!='model._fc.extra_outputs_layer.bias'}
    model_dict.update(pretrained_dict)
    origin_model.load_state_dict(model_dict)
    print('load tusimple pre-trained success')
    
    train_state = None
    if args.resume:
        train_state = get_exp_train_state(exp_root)
    
    #prepare train inference loss
    weights = [10, 2, 1.25]
    weights = 1./ torch.tensor(weights, dtype=torch.float)
    classes_for_imgs = []
    result_train = '/home/jns2szh/code/PolyLaneNet-master/clean_result_train.csv'
    inf_train = pd.read_csv(result_train, index_col=False)
    for i in range(len(inf_train['img'])):
        train_loss = inf_train['valid_loss'][i]
        if train_loss > 0.5 and train_loss < 1.0:
            classes_for_imgs.append(1)
        elif train_loss >= 1.0:
            classes_for_imgs.append(2)
        else:
            classes_for_imgs.append(0)
    samples_weights = weights[classes_for_imgs]

    # Data loader
    '''
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler = sampler,
                                               num_workers=8)
    '''
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    if args.validate:
        val_dataset = cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8)
    # Train regressor
    try:
        model = train(
            model,
            train_loader,
            exp_root,
            cfg,
            val_loader=val_loader if args.validate else None,
            train_state=train_state,
        )
    except KeyboardInterrupt:
        logging.info("Training session terminated.")
