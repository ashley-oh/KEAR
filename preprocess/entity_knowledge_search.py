import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import json
from collections import defaultdict
import re

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

def do_everything(f):
  miss = 0
  for i in tqdm(f, total = len(t)):
    clue = i["inputs"]["clue"]
    inference = i["targets"]["inference"]

    clue_spans = get_spans(clue)
    inference_spans = get_spans(inference)
    
    edges=[]
    for inf in inference_spans:
      if inf in conceptnet.keys():
        temp = conceptnet[inf]
        k = temp.keys()
        for clu in clue_spans:
          try:
            kc = [z for z in k if re.search("umbrella", z)]
      
            for n, s in enumerate(kc):
                    edges.append((inf, k[n], inf[k[n]]))
         
          except:
            miss +=1
            continue
            
    i["relations"] = edges
  return miss

print(do_everything(v))
  
  
