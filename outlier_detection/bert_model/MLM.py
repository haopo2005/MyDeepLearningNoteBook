import sys
sys.path
sys.path.insert(0,'/home/yangjeff/.conda/envs/Trans/lib/python3.6/site-packages')
import torch
from torch.utils.data import DataLoader,Dataset

device=torch.device('cuda')
import os,sys
import pandas as pd
import numpy as np
import re
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
from transformers import RobertaConfig
# from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
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


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass


InputDataClass = NewType("InputDataClass", Any)

@dataclass
class collate_fn(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples) -> Dict[str, torch.Tensor]:
        datas=np.array(examples)#batch is a list,is the index of the input word
        outarr=np.concatenate(datas,axis=0)
        Listtensor=list(torch.from_numpy(outarr).long())
        batch = self._tensorize_batch(Listtensor)
#         print('che:',batch.shape)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "masked_lm_labels": labels}
        else:
            return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
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
def Tok(tokenizer,lines):#sample encode into embedding, each lines is 1x7, one sample has >2000 lines
    batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True)
    examples = batch_encoding["input_ids"]
    assert len(examples)==7# for one sample is 7 features
    
    return examples#torch.tensor(examples[i], dtype=torch.long)
Class mpcDataset(Dataset):#for class training
    def __init__(self,model,data_df,tokenizer,max_len):
        self.token=tokenizer
        self.data=data_df['FilePath']
        self.label=data_df['Label']
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
        toks=np.array(curv,dtype='str').tolist()
        #tokid=self.token.batch_encode_plus(toks, add_special_tokens=True,max_length=self.max_len,pad_to_max_length=True,return_token_type_ids=True)
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
        if single_len>=max_len:
            lines=lines[:max_len-1]
            tmp_mask=np.ones_like(np.arange(max_len))
            mask=torch.tensor(tmp_mask,dtype=torch.long)
            lines=torch.tensor(lines,dtype=torch.long)
            mdout=self.MLM(lines)
        else:
            lines=torch.tensor(lines,dtype=torch.long)
            tmp_mdout=self.MLM(lines)
            dif=max_len-single_len
            padding=torch.zeros((dif,mdout.shape[1]))
            mdout=torch.cat((mdout,padding))
            d1=np.ones(single_len,)
            d2=np.zeros(dif)
            tmp_mask=np.concatenate((d1,d2))
            mask=torch.tensor(tmp_mask,dtype=torch.long)
    
        #token_type_ids=tokid['token_type_ids']
        labs=torch.tensor([int(lab)],dtype=torch.long)
        return {'enc':mdout,'mask':mask,'targets':labs}
        
    def __len__(self):
        return len(self.data)

def collate_fn2(batch):
   
    datas=np.array(batch)#batch is a list
    outarr=np.concatenate(datas,axis=0)
    outtensor=torch.from_numpy(outarr).float()
    return outtensor
class tokDataset(Dataset):#for class training
    def __init__(self,data_df,tokenizer):
        super(tokDataset,self).__init__()
        self.token=tokenizer
        self.data=data_df['FilePath']
        self.count=8
        self.rate=275
            
   def __getitem__(self,idx):
        inf=self.data[idx]
        df_curve=parse_curve(inf)
        df_curve=df_curve.round(5)
        curv=df_curve.values#one sample -1x7 round 2000 lines
#         num=int(len(curv)//self.rate)+1#gap 275
        lines=[]
        check=0
        ids=np.random.choice(len(curv)-1,self.count)

        for chos in ids:
            line=curv[chos]#
            line[line==0.]=0.0
            toks=list(np.array(line,dtype='str'))
            tokid=self.token.encode_plus(toks, add_special_tokens=True)
            tokp=tokid['input_ids']
            if check==1:
                tmp=np.array(tokp)
                try:
                    tp=np.array(tmp,dtype=int)
                except Exception as e:
                    print('err:',e)
                    print('toks:',toks,line)
            lines.append(tokp)
        return lines
    def __len__(self):
        return len(self.data)
Class BERTclass(torch.nn.Module):
    def __init__(self):
        super(BERTclass,self).__init__()
        model_path='/home/yangjeff/MPC2/Token_model/checkpoint-460000-5.35'
        sefl.l1=RobertaModel.from_pretrained(model_path)
        for param in l1.parameters():
            param.requires_grad=False
        self.l2=torch.nn.Dropout(0.3)
        self.l3=torch.nn.Linear(768,168)
    def forward(self,ids):
        emb=self.l1(ids)[0]#input should not padding len=7+2=9
        dropnet=self.l2(emb)
        linet=self.l3(dropnet)
        emb2=emb.view(linet.shape[0],-1)
        return emb2
Class JeffBERT(torch.nn.Module):
    def __init__(self):
        super(JeffBERT,self).__init__()#input_ids=None,inputs_embeds
        config = LongformerConfig(vocab_size=100,num_labels=2,max_length=2560,max_position_embeddings=2560)
        sefl.l1=JeffLongformerForSequenceClassification(config=config)
    def forward(self,embding,att_mask,labels):
        loss,logit=self.l1(inputs_embeds=embding,attention_mask=att_mask,labels=labels)[:2]
        return loss,logit
def train_Class(tr_data,te_data,outmodel):
    bs=64
    pad_len=2560
    EPOCHS=20
    fvoc=open(vocf)
    vlen=len(fvoc.readlines())
    fvoc.close()
    ttk=BertTokenizer.from_pretrained(vocf)
    model=BERTclass()
    model.to(device)
    trdataset=mpcDataset(model,tr_data,ttk,pad_len)
    trainloader = DataLoader(trdataset, batch_size=bs,shuffle=True,drop_last=False)
    tedataset=mpcDataset(model,te_data,ttk,pad_len)
    testloader = DataLoader(tedataset, batch_size=bs,shuffle=True,drop_last=False)
    def train(epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for _,data in enumerate(training_loader, 0):
            embdding = data['enc'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(embdding, mask,targets)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")

        return
    for epoch in range(EPOCHS):
        train(epoch)
    classmodel=JeffBERT()
    classmodel.to(device)
   
    
    training_args = TrainingArguments(
            output_dir=outmodel,#embedding model path
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_device_train_batch_size=bs,
            save_steps=10_000,
            save_total_limit=2,
            
        )

    trainer = Trainer(
        model=classmodel,
        args=training_args,
        
        train_dataset=trdataset,
        eval_dataset=
        data_collator=trainloader,
    )
    trainer.train()
    trainer.save_model(outmodel)
    print('LM train done: ')

    
    
    
def train_MLM(vocf,outmodel,data_df):
    bs=8
    #tokenizer=BertWordPieceTokenizer(vocf)#input vocab.txt
    ttk=BertTokenizer.from_pretrained(vocf)#input vocab.txt
    fvoc=open(vocf)
    vlen=len(fvoc.readlines())
    fvoc.close()
    config=RobertaConfig(vocab_size=vlen,max_position_embeddings=12,num_attention_heads=12, \
                             num_hidden_layers=6,type_vocab_size=1,hidden_size=768)
    model=RobertaForMaskedLM(config=config)
    model.num_parameters()
    
    dataset=tokDataset(data_df,ttk)
#     Data= DataLoader(dataset, batch_size=bs,shuffle=True,drop_last=False,num_workers=0,collate_fn=collate_fn)
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=ttk, mlm=True, mlm_probability=0.15
#     )
   
    data_collator=collate_fn(
        tokenizer=ttk, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
            output_dir=outmodel,#embedding model path
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=bs,
            save_steps=10_000,
            save_total_limit=2,
            
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        
        train_dataset=dataset,
        data_collator=data_collator,
        prediction_loss_only=True
    )
    trainer.train()
    trainer.save_model(outmodel)
    print('LM train done: ')

root='/home/yangjeff/MPC2/tools'
data_df=pd.read_csv('all_dataset.csv')
Token_fold=os.path.join('.','Token_output')
flp=os.path.join(root,'vocab')
vocf='/home/yangjeff/MPC2/Token_model/vocab/all_vocab.txt'
assert os.path.exists(vocf)
if not os.path.exists(Token_fold):
    mk_test(Token_fold)
train_MLM(vocf,Token_fold,data_df)