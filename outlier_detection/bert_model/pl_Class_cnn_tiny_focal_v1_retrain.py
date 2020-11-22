import math
import sys
sys.path.insert(0,'/home/yangjeff/.conda/envs/Trans/lib/python3.6/site-packages')

import torch
from torch.utils.data import DataLoader,Dataset

device=torch.device('cuda')
import datetime
import os,sys
import pandas as pd
import numpy as np
import re
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
from transformers import RobertaConfig
# from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM,RobertaModel
from transformers import BertTokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from abc import ABC, abstractmethod
from transformers import LongformerConfig,JeffLongformerForSequenceClassification
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils import PreTrainedTokenizer
#os.environ['CUDA_LAUNCH_BLOCKING']=1
import torch
import torch.tensor as Tensor
from torch import nn
from torch import optim
import numpy as np
#from models.types_ import *
import os,sys
import pytorch_lightning as pl
from torch.nn import functional as F
#from torchvision import transforms
#import torchvision.utils as vutils
from typing import List
from torch.utils.data import DataLoader,Dataset
import h5py
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging.test_tube import TestTubeLogger # import TestTubeLogger
# For reproducibility
torch.manual_seed(123)
np.random.seed(123)
cudnn.deterministic = True
cudnn.benchmark = False
import torchsnooper
class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv1d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class LogitsLoss(nn.Module):

    def __init__(self,):
        super(LogitsLoss, self).__init__()
        self.prior = [0.9934,0.0066]#0--ok ,1---nok
        self.log_prior = torch.from_numpy(np.log(self.prior))
        

    def forward(self, input, target):
        for _ in range((input.ndim)-1):
            self.log_prior=self.log_prior.unsqueeze(0)
        y_pred=input+self.log_prior.to(input.device)
        crition=nn.CrossEntropyLoss()
        loss = crition(y_pred,target)
        return loss
class JeffBERT(torch.nn.Module):
    def __init__(self,maxlen,premd,mk=0):
        super(JeffBERT, self).__init__()
        self.num_labels=2
        num_filters=100
        self.embedding_size = 768
        model_path='/home/yangjeff/MPC2/Token_model/checkpoint-80000-5.33'# use tiny roberta MLM model
        self.MLM=RobertaModel.from_pretrained(model_path)
        
        for param in self.MLM.parameters():
            param.requires_grad=False
        self.MLM=self.MLM.to(device)
        self.conv_region = nn.Conv2d(1, num_filters, (3, self.embedding_size), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, self.num_labels)
        self.conv_region=self.conv_region.to(device)
        self.conv=self.conv.to(device)
        self.max_pool=self.max_pool.to(device)
        self.padding1=self.padding1.to(device)
        self.padding2=self.padding2.to(device)
        self.relu=self.relu.to(device)
        self.fc=self.fc.to(device)
        
        
    def forward(self,input):
        ids=input['ids']
        ids=ids.to(self.MLM.device)
        #print('JeffBert:',ids[0].shape,self.MLM(ids[0])[0].shape,ids.shape)
        new_emb0=torch.stack([self.MLM(item)[0] for item in ids])# x=[batch_size,seq_len,embedding_dim]
        new_emb=torch.mean(new_emb0,-2)
        #print('jeff2:',new_emb.shape,xx_emb.shape)
        
        new_emb=new_emb.to(self.MLM.device)
        #assert 0>1
        x=new_emb
        labels=input['target']
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        x = self.conv_region(x)  # x = [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)  # [batch_size, num_filters,1,1]
        x = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x)  # [batch_size, 1]
        return [x,labels]

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
    def calcuate_accu(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        #print('loss_function:',args)
        resout = args[0]
        target=args[1]
        
        big_val, big_idx = torch.max(resout.data, dim=1)
        #print('xx:',big_val,big_idx)
        n_correct=(big_idx==target).sum().item()
        #Loss=LogitsLoss()
        Loss=MultiFocalLoss(num_class=self.num_labels)
        real_loss =Loss(resout.view(-1, self.num_labels),target.view(-1))
        return {'loss': real_loss,'acc':n_correct
                }
def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper
def parse_curve(filepath):

    """
    Parses the curve into a pandas dataframe

    """
    df = pd.read_csv(

        filepath,

        skiprows=33,

        delimiter=';',

        dtype=np.float,

        names=['WegSpd1','KraftSpd1','K7_1_TL','K7_2_TR','K7_3_BL','K7_4_BR','K3m_MC'],

        usecols=range(7)

    )

    df.name = os.path.basename(filepath)
    return df
class BertWrap(pl.LightningModule):

    def __init__(self,
                 bert_model: nn.Module,
                 params: dict,
                mk) -> None:
        super(BertWrap, self).__init__()
        self.model = bert_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.pad_len=self.params['max_length']# times of 512
        self.testmk=mk
        #Bfile = os.path.join('sample_submission_res.txt')
        #self.bcsv=open(Bfile,"a")
        vocf='/home/yangjeff/MPC2/Token_model/vocab/all_vocab.txt'
        fvoc=open(vocf)
        vlen=len(fvoc.readlines())
        fvoc.close()
        trdata_path='/home/yangjeff/MPC2/Train_data.csv'
        tedata_path='/home/yangjeff/MPC2/Eval_data.csv'
        added_path='/home/yangjeff/MPC2/added_trainmission.csv'
        if self.testmk==1:
            
            tmt=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            tname='MPC2'+tmt+'.txt'
            Dfile = os.path.join('/home/yangjeff/MPC2', tname)
            self.dcsv=open(Dfile,"a")
        self.ttk=BertTokenizer.from_pretrained(vocf)

        tmpdf=pd.read_csv(trdata_path)
        
        
        self.te_data=pd.read_csv(tedata_path)
        self.tr_data=pd.concat((tmpdf,self.te_data))
        #print('chec0:',self.tr_data.shape,tmpdf.shape,self.te_data.shape)
        if os.path.exists(added_path):
            added_data=pd.read_csv(added_path)
            self.tr_data=pd.concat((self.tr_data,added_data))
            #print('chec01:',self.tr_data.shape,added_data.shape)
        self.tr_data=self.tr_data.reset_index(drop=True)
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    

    class mpcDataset(Dataset):#for class training
        def __init__(self,data_df,tokenizer,max_len):
            self.token=tokenizer
            self.data=data_df['FilePath']
            self.label=data_df['Label']
            self.max_len=max_len
            
        def __getitem__(self,idx):
            inf=self.data[idx]
            lab=self.label[idx]
            df_curve=parse_curve(inf)
            df_curve=df_curve.round(5)
            curv=df_curve.values
            curv[curv==0.]=0.0
            check=4
            indx=np.linspace(0,len(curv)-1,num=self.max_len,endpoint=False,retstep=False, dtype=int)
            tmp_curv=curv[indx]
            toks=np.array(tmp_curv,dtype='str').tolist()
           
            tokid=self.token.batch_encode_plus(toks, add_special_tokens=False)
            lines=tokid['input_ids']
            if check<4:
                tmp=np.array(lines)
                try:
                    tp=np.array(tmp,dtype=int)
                except Exception as e:
                    print('err:',e)
                    print('toks:',toks)
                check+=1
            single_len=len(lines)
            assert single_len>=self.max_len
                #print('max:',single_len)
            lines=lines[:self.max_len]
            tmp_mask=np.ones_like(np.arange(self.max_len))
            mask=torch.tensor(tmp_mask,dtype=torch.long)
            lines_tensor=torch.tensor(lines,dtype=torch.long)
            labs=torch.tensor([int(lab)],dtype=torch.long)
            #print('mpcdataset:',mdout.shape,mask.shape,labs.shape)
            return {'ids':lines_tensor,'mask':mask,'target':labs}
        def __len__(self):
            return len(self.data)
    class test_mpcDataset(Dataset):#for class training
        def __init__(self,tokenizer,max_len):
            data_path='/home/yangjeff/MPC2/sample_submission_sub.csv'
            data_df=pd.read_csv(data_path)
            self.token=tokenizer
            self.data=data_df['FilePath']
            self.max_len=max_len
            self.root='/home/yangjeff/MPC2/tools'
            self.lab=data_df['Decision']
        def __getitem__(self,idx):
            inf=self.data[idx]
            xnf=os.path.join(self.root,inf.split('/')[0],inf)
            df_curve=parse_curve(xnf)
            df_curve=df_curve.round(5)
            curv=df_curve.values
            curv[curv==0.]=0.0
            check=4
            indx=np.linspace(0,len(curv)-1,num=self.max_len,endpoint=False,retstep=False, dtype=int)
            tmp_curv=curv[indx]
            toks=np.array(tmp_curv,dtype='str').tolist()
           
            tokid=self.token.batch_encode_plus(toks, add_special_tokens=False)
            lines=tokid['input_ids']
            if check<4:
                tmp=np.array(lines)
                try:
                    tp=np.array(tmp,dtype=int)
                except Exception as e:
                    print('err:',e)
                    print('toks:',toks)
                check+=1
            single_len=len(lines)
            assert single_len>=self.max_len
                #print('max:',single_len)
            lines=lines[:self.max_len]
            tmp_mask=np.ones_like(np.arange(self.max_len))
            mask=torch.tensor(tmp_mask,dtype=torch.long)
            lines_tensor=torch.tensor(lines,dtype=torch.long)
            
            
            #print('mpcdataset:',mdout.shape,mask.shape,labs.shape)
            return {'ids':lines_tensor,'mask':mask,'target':inf}
        def __len__(self):
            return len(self.data)
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    def test_step(self, batch, batch_idx):
        results = self.forward(batch)
        #print('results:',results)
        if self.testmk==1:
            resout = results[0]
            target=results[1]
            big_val, big_idx = torch.max(resout.data, dim=1)
            idx=big_idx.cpu().numpy().tolist()
            idv=resout.cpu().numpy().tolist()
            for i in range(len(idx)):
                a1=int(idx[i])
                b1=idv[i]
                til=target[i]
                xg0=np.hstack((a1,b1))
                xg=[til]+b1+[a1]
                xg2=[str(x) for x in xg]
                cont=','.join(xg2)+'\n'
                self.dcsv.write(cont)
            #print('cc:',idx,idv)
            self.dcsv.flush()
        return 1
    def test_end(self, outputs):
        self.dcsv.close()
        tmt=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        print('test done:',str(tmt))
        
        return  {'test_loss': str(tmt)}
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
      
        #print('jeff1:',batch)
        self.curr_device = batch['target'].device
        results = self.forward(batch)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items() if key!='acc'})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
       # print('jeff2:',batch)
        real_img = batch
        self.curr_device = real_img['target'].device
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        '''
        i=0
        for x in outputs:
            if i<1:
                print(x)
                print(type(x))
                print(type(x['acc']))
                print(x['acc'])
                i+=1
        '''
        tensorboard_logs={'val_Loss': avg_loss}
        acc= np.array([x['acc'] for x in outputs]).sum()#torch.stack([x['acc'] for x in outputs]).sum()
        print('in test all right:',acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

#add annealing LR
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'],
                              amsgrad=True)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
            
            if self.params['anneal'] is not None:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optims[0],
                                                             T_max=5)
                scheds.append(scheduler)

                return optims, scheds
        except:
            return optims
    #add warm-up learning rate
    def optimizer_step(self,current_epoch,batch_idx,optimizer,optimizer_idx,second_order_closure=None):
        if self.trainer.global_step<500:
            lr_scale=min(1,float(self.trainer.global_step+1)/500.)
            for pg in optimizer.param_groups:
                pg['lr']=lr_scale*self.params['LR']
        #updata params in default function
        optimizer.step()
        optimizer.zero_grad()
        
    @data_loader
    def test_dataloader(self):
        test_dataset = self.test_mpcDataset(self.ttk,self.pad_len)
        self.testdataloader =DataLoader(test_dataset, batch_size=self.params['batch_size'], shuffle=False)

        return self.testdataloader 
    @data_loader
    def train_dataloader(self):
        tr_dataset=self.mpcDataset(self.tr_data,self.ttk,self.pad_len)
    
        self.trainloader = DataLoader(tr_dataset, batch_size=self.params['batch_size'],shuffle=True,drop_last=True)
        self.num_train_imgs = len(self.trainloader)
        return self.trainloader
   
    @data_loader
    def val_dataloader(self):
        te_dataset = self.mpcDataset(self.te_data,self.ttk,self.pad_len)
        self.sample_dataloader =  DataLoader(te_dataset, batch_size=self.params['batch_size'],shuffle=True,drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

def mk_test(testdir):#clean the folder
    
    if os.path.isdir(testdir):
        rt1_lst=os.listdir(testdir)
        up,cu=os.path.split(testdir)

        if len(rt1_lst)>0:
            for cell in rt1_lst:
                pth=os.path.join(testdir,cell)
                if os.path.isfile(pth):
                    os.remove(pth)
                if os.path.isdir(pth):
                    shutil.rmtree(pth)
    else:
        os.mkdir(testdir)
                                                                                                 
def infer_main(maxlen=512):
    predict_path='/home/yangjeff/MPC2/Predict_data.csv'
    test_data=pd.read_csv(predict_path)
    MLMmodel=MLMclass()
    pad_len=maxlen
    bs=4
    vocf='/home/yangjeff/MPC2/Token_model/vocab/all_vocab.txt'
    ttk=BertTokenizer.from_pretrained(vocf)
    class Test_Dataset(Dataset):#for class training
        def __init__(self,model,data_df,tokenizer,max_len):
            self.token=tokenizer
            self.data=data_df['FilePath']
            self.max_len=max_len
            self.MLM=model
        def __getitem__(self,idx):
            inf=self.data[idx]
            lab=self.label[idx]
            df_curve=parse_curve(inf)
            df_curve=df_curve.round(5)
            curv=df_curve.values
            curv[curv==0.]=0.0
            check=1
            indx=np.linspace(0,len(curv)-1,num=self.max_len,endpoint=False,retstep=False, dtype=int)
            tmp_curv=curv[indx]
            toks=np.array(tmp_curv,dtype='str').tolist()

            tokid=self.token.batch_encode_plus(toks, add_special_tokens=False)
            lines=tokid['input_ids']
            if check<1:
                tmp=np.array(lines)
                try:
                    tp=np.array(tmp,dtype=int)
                except Exception as e:
                    print('err:',e)
                    print('toks:',toks)
                check+=1
            single_len=len(lines)
            if single_len>=self.max_len:
                #print('max:',single_len)
                lines=lines[:self.max_len]
                tmp_mask=np.ones_like(np.arange(self.max_len))
                mask=torch.tensor(tmp_mask,dtype=torch.long)
                lines_tensor=torch.tensor(lines,dtype=torch.long)
                mdout=self.MLM(lines_tensor)
            else:
                lines_tensor=torch.tensor(lines,dtype=torch.long)
                #print('cc:',lines_tensor.device,lines_tensor.shape)
                tmp_mdout=self.MLM(lines_tensor)
                dif=self.max_len-single_len

                padding=torch.zeros((dif,tmp_mdout.shape[1]))
                padding=padding.to(tmp_mdout.device)
                mdout=torch.cat((tmp_mdout,padding))
                d1=np.ones(single_len)
                d2=np.zeros(dif)
                tmp_mask=np.concatenate((d1,d2))
                mask=torch.tensor(tmp_mask,dtype=torch.long)


            return {'enc':mdout,'mask':mask}
        def __len__(self):
            return len(self.data)

    test_dataset = Test_Dataset(MLMmodel,test_data,ttk,pad_len)
    Testdataloader =DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return self.Testdataloader 
if __name__ == "__main__":
   # tr=pd.read_csv('Train_data.csv')
    #te=pd.read_csv('Eval_data.csv')
    #vocf='/home/yangjeff/MPC2/Token_model/vocab/all_vocab.txt'
    #fvoc=open(vocf)
    #vlen=len(fvoc.readlines())
    #fvoc.close()
    maxlen=512*2#3072
    pretrained=[]
    model=JeffBERT(maxlen,pretrained)
    
    param={'dataset':'celeba','max_length':maxlen,'batch_size':128,'LR':0.00001,'weight_decay':0.0,'scheduler_gamma':0.1,'anneal':None}
    experiment = BertWrap(model,param,mk=1)# mk=1 run test infer
    lgpath=os.path.join('.','Logs')
    wgpath=os.path.join('.','Weights')
    if not os.path.exists(lgpath):
        mk_test(lgpath)
    if not os.path.exists(wgpath):
        mk_test(wgpath)
    tt_logger = TestTubeLogger(
                save_dir='./Logs',
                name='VWVAE',
                debug=False,
                create_git_tag=False,
            )
  
    #trainer=Trainer(gpus='-1',max_epochs=1,logger=tt_logger,weights_save_path=wgpath,gradient_clip=0.5,gradient_clip_val=0.5)#,accumulate_grad_batches=32)
    trainer=Trainer(gpus='-1',max_epochs=1,logger=tt_logger,weights_save_path=wgpath,resume_from_checkpoint='/home/yangjeff/MPC2/Train_code/VWVAE/version_81/checkpoints/_ckpt_epoch_0.ckpt')#73 loss=0.02 new81 for model3 
    #print('j0')
    trainer.fit(experiment)
    #print('j1')
    trainer.test()
    '''
    # run test from a loaded model
    #model = LightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
    #trainer = Trainer()
    #trainer.test(model)
    '''
