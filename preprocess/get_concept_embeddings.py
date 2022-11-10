import json
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

conceptnet_path = "../data/kear/filtered_edges.json"

with open(conceptnet_path) as f:
  x = json.load(f)
  
concept =[]
for i in x:
  #for j in x[i]:
  #  concept.append(j)
  concept.append(i)
  
concept = list(set(concept))

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(concept)

df = pd.DataFrame(list(zip(concept, embeddings.tolist())), columns = ["word", "emb"])

df.to_csv("concept_embeddings_filtered.csv", sep = "\t")

#concept_embedding.csv - original
#concept_embedding_filtered.csv -filtered to only include keys