import json
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from collections import defaultdict

conceptnet_path = "../data/kear/filtered_edges.json"

with open(conceptnet_path) as f:
  x = json.load(f)
  
count_dict = defaultdict(int)
concept =[]

for i in x:
  for j in x[i]:
    concept.append(j)
    count_dict[j]+=1
  concept.append(i)
  counts_dict[i]+=1
  
concept = list(set(concept))

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(concept)

counts = [count_dict[i] for i in concept]

df = pd.DataFrame(list(zip(concept, embeddings.tolist(), counts)), columns = ["word", "emb", "count"])

df.to_csv("concept_embeddings_counts.csv", sep = "\t")

#concept_embedding.csv - original
#concept_embedding_filtered.csv -filtered to only include keys
#"concept_embeddings_counts.csv" - original + counts of how many times word appears in conceptnet
