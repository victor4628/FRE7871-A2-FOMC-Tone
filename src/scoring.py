"""Published policy rules and two distinct sentence-classification models."""
from __future__ import annotations
import hashlib
import json
import os
import numpy as np
import pandas as pd
from .config import INTERIM_DIR, ROOT
from .preprocess import VERSION, prepare_documents

# Shah et al. (ACL 2023), code_model/rule_based.py, CC BY-NC 4.0.
# Original substring matching, dovish precedence and "decelarate" spelling retained.
A1 = ['inflation expectation','interest rate','bank rate','fund rate','price','economic activity','inflation','employment']
A2 = ['anchor','cut','subdue','decline','decrease','reduce','low','drop','fall','fell','decelarate','slow','pause','pausing','stable','non-accelerating','downward','tighten']
B1 = ['unemployment','growth','exchange rate','productivity','deficit','demand','job market','monetary policy']
B2 = ['ease','easing','rise','rising','increase','expand','improve','strong','upward','raise','high','rapid']
C = ["weren't",'were not',"wasn't",'was not','did not',"didn't",'do not',"don't",'will not',"won't"]
MODELS = {'finbert': ('ProsusAI/finbert','4556d13015211d73dccd3fdd39d39232506f3e43'),
          'roberta': ('gtfintechlab/FOMC-RoBERTa', None)}

def rule_sentence(sentence: str) -> dict:
    s = sentence.lower().replace('’', "'")
    hits = {k:[w for w in words if w in s] for k,words in [('A1',A1),('A2',A2),('B1',B1),('B2',B2),('C',C)]}
    dove = bool((hits['A1'] and hits['A2']) or (hits['B1'] and hits['B2']))
    hawk = bool((hits['A1'] and hits['B2']) or (hits['B1'] and hits['A2']))
    label = 0 if dove else 1 if hawk else 2
    if label != 2 and hits['C']: label = 1-label
    return {'rule_label':label, 'rule_target':bool(hits['A1'] or hits['B1']),
            'rule_conflict':dove and hawk, 'rule_negated':label != 2 and bool(hits['C']),
            'rule_matches':json.dumps(hits,ensure_ascii=False)}

def wordlist_score(text: str) -> dict:
    from .preprocess import SPLITTER
    rows = [rule_sentence(s) for s in SPLITTER.tokenize(text)]
    targets = [r for r in rows if r['rule_target']]
    h,d = sum(r['rule_label']==1 for r in targets),sum(r['rule_label']==0 for r in targets)
    return {'hawk_hits':h,'dove_hits':d,'wordlist_score':(h-d)/len(targets) if targets else np.nan}

class SentenceClassifier:
    def __init__(self, method: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch, self.method = torch, method
        torch.set_num_threads(4)
        model_id, revision = MODELS[method]
        kwargs = {'cache_dir':str(ROOT/'data/hf_cache/hub')}
        if revision: kwargs['revision'] = revision
        override = os.environ.get('FOMC_MODEL_PATH') if method == 'roberta' else None
        if override: model_id, kwargs = override, {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', None) or revision or str(model_id)
        labels = {int(k):str(v).lower() for k,v in self.model.config.id2label.items()}
        if method == 'finbert':
            if set(labels.values()) != {'positive','negative','neutral'}: raise ValueError(f'Unexpected FinBERT labels: {labels}')
            self.indices = [next(k for k,v in labels.items() if v==x) for x in ['positive','negative','neutral']]
        else:
            if labels not in ({0:'label_0',1:'label_1',2:'label_2'}, {0:'dovish',1:'hawkish',2:'neutral'}): raise ValueError(f'Unexpected FOMC labels: {labels}')
            self.indices = [0,1,2]
        self.metadata = {'model':model_id,'revision':self.revision,'labels':labels,'device':self.device,
                         'long_sentence_rule':'nonoverlapping token chunks; token-weighted mean probabilities; equal sentence weights'}

    def predict(self, texts: list[str], batch_size: int = 48) -> tuple[np.ndarray,np.ndarray]:
        torch = self.torch
        limit = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings, 512)
        limit -= self.tokenizer.num_special_tokens_to_add(pair=False)
        pieces, owners, weights = [], [], []
        for i,text in enumerate(texts):
            ids = self.tokenizer.encode(text,add_special_tokens=False)
            for start in range(0,len(ids),limit):
                part = ids[start:start+limit]
                pieces.append(self.tokenizer.prepare_for_model(part,add_special_tokens=True,return_attention_mask=True))
                owners.append(i); weights.append(len(part))
        result = np.zeros((len(texts),3)); totals = np.zeros(len(texts)); chunks = np.zeros(len(texts),dtype=int)
        with torch.inference_mode():
            for start in range(0,len(pieces),batch_size):
                batch = self.tokenizer.pad(pieces[start:start+batch_size],padding=True,return_tensors='pt').to(self.device)
                prob = torch.softmax(self.model(**batch).logits,dim=-1).float().cpu().numpy()[:,self.indices]
                for j,p in enumerate(prob,start):
                    i = owners[j]; result[i] += p*weights[j]; totals[i] += weights[j]; chunks[i] += 1
        return result / totals[:,None], chunks

def score_documents(documents: pd.DataFrame, refresh=False, methods=('finbert','roberta')) -> pd.DataFrame:
    docs,sents = prepare_documents(documents)
    rules = pd.DataFrame([rule_sentence(s) for s in sents.sentence])
    sents = pd.concat([sents,rules],axis=1)
    meta = {'selection_version':VERSION,'rule_source_revision':'98646987452b326507479cf641571b33814bb73f'}
    for method in methods:
        scorer = None
        labels = ['positive','negative','neutral'] if method == 'finbert' else ['dovish','hawkish','neutral']
        cols = [method+'_p_'+x for x in labels]
        for _,doc in docs.iterrows():
            doc_idx = sents.index[sents.document_id==doc.document_id]
            idx = doc_idx if method == 'finbert' else sents.index[(sents.document_id==doc.document_id) & sents.rule_target]
            texts = sents.loc[idx,'sentence'].tolist()
            if not texts:
                continue
            sample = 'full_prose' if method == 'finbert' else 'policy_target'
            identity=method+sample+doc.selected_sha256+str(MODELS[method])+os.environ.get('FOMC_MODEL_PATH','')
            key = hashlib.sha256((VERSION+identity).encode()).hexdigest()
            cache = INTERIM_DIR/'sentence_cache'/f'{key}.json'
            legacy_key=hashlib.sha256(('selection-v2-20260913'+identity).encode()).hexdigest()
            legacy_cache=INTERIM_DIR/'sentence_cache'/f'{legacy_key}.json'
            if not cache.exists() and legacy_cache.exists() and not refresh:
                cache.parent.mkdir(parents=True,exist_ok=True)
                cache.write_bytes(legacy_cache.read_bytes())
            if cache.exists() and not refresh:
                payload = json.loads(cache.read_text()); prob=np.array(payload['prob']); chunks=payload['chunks']
            else:
                if scorer is None: scorer = SentenceClassifier(method)
                prob,chunks = scorer.predict(texts)
                payload={'prob':prob.tolist(),'chunks':chunks.tolist(),'metadata':scorer.metadata}
                cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(payload))
            sents.loc[idx,cols] = prob
            sents.loc[idx,method+'_chunks'] = chunks
            meta[method]=payload['metadata']
            print(f'{method}: {doc.release_date} {doc.subtype}',flush=True)
        if scorer is not None:
            del scorer
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[]
    for _,doc in docs.iterrows():
        eligible=sents[sents.document_id==doc.document_id]; target=eligible[eligible.rule_target]
        row=doc.to_dict();n=len(target)
        row.update(target_sentences=n,coverage=n/len(eligible) if len(eligible) else np.nan,
                   hawk_hits=int((target.rule_label==1).sum()),dove_hits=int((target.rule_label==0).sum()),rule_conflicts=int(target.rule_conflict.sum()))
        row['wordlist_score']=(row['hawk_hits']-row['dove_hits'])/n if n else np.nan
        if 'finbert' in methods:
            row['finbert_sentiment']=(target.finbert_p_positive-target.finbert_p_negative).mean()
            row['finbert_positive']=target.finbert_p_positive.mean()
            row['finbert_full_prose']=(eligible.finbert_p_positive-eligible.finbert_p_negative).mean()
        if 'roberta' in methods:
            probs=target[['roberta_p_dovish','roberta_p_hawkish','roberta_p_neutral']].to_numpy();lab=probs.argmax(axis=1)
            row['roberta_score']=float(((lab==1).sum()-(lab==0).sum())/n) if n else np.nan
            row['roberta_probability_score']=float((probs[:,1]-probs[:,0]).mean()) if n else np.nan
        rows.append(row)
    result=pd.DataFrame(rows).sort_values(['release_date','subtype'])
    for score in ['wordlist_score','roberta_score','finbert_sentiment']:
        if score in result: result[score+'_change']=result.groupby('subtype')[score].diff()
    suffix='' if 'roberta' in methods else '_partial'
    result.to_csv(INTERIM_DIR/f'document_scores{suffix}.csv',index=False)
    sents.to_csv(INTERIM_DIR/f'sentence_scores{suffix}.csv',index=False)
    (INTERIM_DIR/f'scoring_metadata{suffix}.json').write_text(json.dumps(meta,indent=2))
    return result
