import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import json

#set up spacy
nlp = spacy.load("en_core_web_sm")

#set up matcher
pattern = [[{'DEP': 'compound', 'OP': '?'}, {'POS': 'NOUN'}], [{'DEP': 'compound', 'OP': '?'}, {'POS': 'PROPN'}], [{'POS': 'VERB'}, {'DEP': 'acomp', 'OP': '?'}]]
matcher = Matcher(nlp.vocab)
matcher.add("compounds", pattern)

train_path = "/capstone/sherlock_train_v1_1.json"
val_path = "/capstone/sherlock_val_with_split_idxs_v1_1.json"

with open(train_path) as f:
  t = json.load(f)
  
with open(val_path) as f:
  v = json.load(f)

def get_spans(text):
  doc = nlp(text)
  matches = matcher(doc)
  spans = []
  for match_id, start, end in matches:
     span = doc[start:end] 
     if end - start >1:
      spans.append(span.text.replace(" ", "_")
     else:
      if span[0].dep_ not in ["compound", "acomp"]:
          spans.append(span[0].lemma_)
        
  return spans  
  
for i in tqdm(t, total = len(t)):
  clue = i["inputs"]["clue"]
  inference = i["targets"]["inference"]
  
  clue_spans = get_spans(clue)
  inference_spans
  
  
  
