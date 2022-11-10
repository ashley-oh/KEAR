import json
import pandas as pd

from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import ast
import numpy as np
import sys 

graded_weights = bool(sys.argv[1])
filepath = sys.argv[1]

def stuff(x):
     for i in tqdm(x, total = len(x)):
             clue = i["inputs"]["clue"]
             inference = i["targets"]["inference"]
             clue_emb = torch.from_numpy(np.reshape(model.encode(clue), (1,-1)))
             inf_emb = torch.from_numpy(np.reshape(model.encode(inference), (1,-1)))
             clue_cos = cos(clue_emb, emb) *weights
             inf_cos = cos(inf_emb, emb) *weights
             i["inputs"]["clue_context"]=df["word"][torch.topk(clue_cos, 5).indices.tolist()]
             i["targets"]["inference_context"]=df["word"][torch.topk(clue_cos, 5).indices.tolist()]


df = pd.read_csv("concept_embeddings_counts.csv", sep = "\t")          
#df = df[df["count"]>1]
emb = [ast.literal_eval(i) for i in tqdm(df["emb"].to_list(), total = len(df["emb"].to_list())]
emb = torch.tensor(emb)

if graded_weights:                                        
     count_to_weight = {1:.97, 2:.98, 3:99 }
     weights = [count_to_weight[i] if i <4 else 1 for i in df["count"].tolist()]
     weights = torch.tensor(weights)

train_path = "/capstone/sherlock_train_v1_1.json"
val_path = "/capstone/sherlock_val_with_split_idxs_v1_1.json"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cos = torch.nn.CosineSimilarity(dim=1)

with open(train_path) as f:
   t = json.load(f)
    
with open(val_path) as f:
    v = json.load(f)

stuff(v)
with open("val_"+filepath) as f:
     json.dump(v, f)
                                         
stuff(t)
with open("train_"+filepath) as f:
     json.dump(v, f)

  
  
              
