import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import json
from collections import defaultdict
import re

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from torch.multiprocessing import Pool,set_start_method
#import multiprocessing as mp


#set up spacy
nlp = spacy.load("en_core_web_sm")

#set up matcher
pattern = [[{'DEP': 'compound', 'OP': '?'}, {'POS': 'NOUN'}], [{'DEP': 'compound', 'OP': '?'}, {'POS': 'PROPN'}], [{'POS': 'VERB'}, {'DEP': 'acomp', 'OP': '?'}]]
matcher = Matcher(nlp.vocab)
matcher.add("compounds", pattern)

train_path = "/capstone/sherlock_train_v1_1.json"
val_path = "/capstone/sherlock_val_with_split_idxs_v1_1.json"


#get conceptnet
conceptnet={}

with open('../data/kear/conceptnet.en.csv', encoding='utf-8') as cpnet:
     for line in cpnet:
             ls = line.strip().split('\t')
             rel = ls[0]
             subj = ls[1]
             obj = ls[2]
             weight = float(ls[3])
             if subj not in conceptnet:
                     conceptnet[subj] = {}
             if obj not in conceptnet[subj]:
                     conceptnet[subj][obj] = defaultdict(int)
             conceptnet[subj][obj][rel] = max(conceptnet[subj][obj][rel], weight)


with open(train_path) as f:
  t = json.load(f)
  
with open(val_path) as f:
  v = json.load(f)

def get_spans(text):
  doc = nlp(text)
  matches = matcher(doc)
  spans =[]
  for match_id, start, end in matches:
     span = doc[start:end] 
     if end - start >1:
      spans.append(span.text.replace(" ", "_"))
     else:
      if span[0].dep_ not in ["compound", "acomp"] and span[0].lemma_ != "be":
          spans.append(span[0].lemma_)
        
  return spans  

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cos = torch.nn.CosineSimilarity(dim=1)

def do_everything(i):
  #for i in tqdm(f, total = len(f)):
    clue = i["inputs"]["clue"]
    inference = i["targets"]["inference"]

    clue_spans = get_spans(clue)
    inference_spans = get_spans(inference)
    
    edges=[]
    for inf in inference_spans:

      if inf in conceptnet.keys():
        temp = conceptnet[inf]
        k = list(temp.keys())
        k_emb = torch.from_numpy(model.encode(k))
        for clu in clue_spans:
            try: 
                edges.append((inf, list(conceptnet[inf][clu].keys())[0], clu))
            #kc = [z for z in k if re.search("(_|\b)"+clu+"(_|\b)", z)]
            except KeyError:
                clue_emb = torch.from_numpy(np.reshape(model.encode(clu), (1,-1)))
            #for s in kc:
            #        edges.append((inf, list(conceptnet[inf][s].keys())[0], s ))
                clue_cos = cos(clue_emb, k_emb)
                c= torch.argmax(clue_cos).item()
           # for c in torch.topk(clue_cos, 1).indices.tolist():
            
                edges.append((inf, list(conceptnet[inf][k[c]].keys())[0], k[c] ))
    
            
      i["relations"] = edges


if __name__ == '__main__':

    set_start_method("spawn")

    func_name = do_everything
    val_out=[]
    train_out=[]
    with Pool(8) as p:
        val_out=list(tqdm(p.imap(func_name, v, chunksize=8), total = len(v)))
    with Pool(8) as p:
        train_out=list(tqdm(p.imap(func_name, t, chunksize=8), total = len(t)))
#do_everything(v)

#do_everything(t)

    with open("val_entity_relation_cos.json", "w") as f:
         json.dump(val_out, f)
                                         

    with open("train_entity_relation_cos.json", "w") as f:
         json.dump(train_out, f)

