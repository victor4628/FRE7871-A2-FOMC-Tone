"""Contingency reproduction trained only on the authors' pre-2020 labels."""
from __future__ import annotations
import json,random,time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from .config import ROOT

SEED=5768;BASE='FacebookAI/roberta-large';MAX_LENGTH=256;BATCH=4;ACCUM=4;EPOCHS=8;PATIENCE=2;LR=1e-5
AUTHOR_REVISION='98646987452b326507479cf641571b33814bb73f'
AUTHOR_RAW=f'https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/{AUTHOR_REVISION}/look_ahead_bias'
DATA_URLS={name:f'{AUTHOR_RAW}/{name}' for name in ('1996-2019-train.xlsx','2020-2022-test.xlsx')}

def ensure_author_data(source:Path)->None:
    """Fetch the authors' fixed temporal split when it is not already cached."""
    source.mkdir(parents=True,exist_ok=True)
    for name,url in DATA_URLS.items():
        target=source/name
        if target.exists():continue
        response=requests.get(url,timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)

class Rows(Dataset):
    def __init__(self,frame,tokenizer):self.text=frame.sentence.astype(str).tolist();self.label=frame.label.astype(int).tolist();self.tok=tokenizer
    def __len__(self):return len(self.text)
    def __getitem__(self,i):
        x=self.tok(self.text[i],truncation=True,max_length=MAX_LENGTH)
        x['labels']=self.label[i];return x

def evaluate(model,loader,device):
    model.eval();losses=[];true=[];pred=[]
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()};out=model(**batch)
            losses.append(float(out.loss));true.extend(batch['labels'].cpu().tolist());pred.extend(out.logits.argmax(1).cpu().tolist())
    report=classification_report(true,pred,output_dict=True,zero_division=0)
    return float(np.mean(losses)),report,true,pred

def train():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    source=ROOT/'data/training';ensure_author_data(source)
    dest=ROOT/'data/models/fomc-roberta-temporal';dest.mkdir(parents=True,exist_ok=True)
    frame=pd.read_excel(source/'1996-2019-train.xlsx').dropna(subset=['sentence','label'])
    train,val=train_test_split(frame,test_size=.15,random_state=SEED,stratify=frame.label)
    test=pd.read_excel(source/'2020-2022-test.xlsx').dropna(subset=['sentence','label'])
    tokenizer=AutoTokenizer.from_pretrained(BASE,cache_dir=str(ROOT/'data/hf_cache/hub'))
    model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=3,cache_dir=str(ROOT/'data/hf_cache/hub'),
        id2label={0:'Dovish',1:'Hawkish',2:'Neutral'},label2id={'Dovish':0,'Hawkish':1,'Neutral':2})
    model.gradient_checkpointing_enable();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    collate=lambda batch:tokenizer.pad(batch,padding=True,return_tensors='pt')
    g=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(Rows(train,tokenizer),batch_size=BATCH,shuffle=True,generator=g,collate_fn=collate)
    val_loader=DataLoader(Rows(val,tokenizer),batch_size=BATCH*2,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(Rows(test,tokenizer),batch_size=BATCH*2,shuffle=False,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf');bad=0;history=[];start=time.time()
    for epoch in range(EPOCHS):
        model.train();optimizer.zero_grad(set_to_none=True);running=[]
        for step,batch in enumerate(train_loader):
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.amp.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                loss=model(**batch).loss/ACCUM
            scaler.scale(loss).backward();running.append(float(loss)*ACCUM)
            if (step+1)%ACCUM==0 or step+1==len(train_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        val_loss,val_report,_,_=evaluate(model,val_loader,device)
        row={'epoch':epoch+1,'train_loss':float(np.mean(running)),'val_loss':val_loss,
             'val_weighted_f1':val_report['weighted avg']['f1-score']};history.append(row);print(row,flush=True)
        if val_loss<best:
            best=val_loss;bad=0;model.save_pretrained(dest);tokenizer.save_pretrained(dest)
        else:
            bad+=1
            if bad>=PATIENCE:break
    model=AutoModelForSequenceClassification.from_pretrained(dest).to(device)
    test_loss,test_report,true,pred=evaluate(model,test_loader,device)
    meta={'status':'contingency reproduction; not the gated official checkpoint','base_model':BASE,
          'author_repository_revision':AUTHOR_REVISION,
          'training_years':'1996-2019','test_years':'2020-2022','seed':SEED,'max_length':MAX_LENGTH,
          'batch_size':BATCH,'gradient_accumulation':ACCUM,'learning_rate':LR,'history':history,
          'test_loss':test_loss,'test_report':test_report,'elapsed_seconds':time.time()-start,
          'train_n':len(train),'validation_n':len(val),'test_n':len(test)}
    (dest/'training_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
    return dest
if __name__=='__main__':train()
