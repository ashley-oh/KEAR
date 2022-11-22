import json
from tqdm import tqdm
with open("/capstone/capstone-localization/val_localization/val_localization_instances.json") as f:
    val = json.load(f)
    
with open("../data/sherlock_new/val_concept.json") as f:
    concepts = json.load(f) 
    
img_to_clue = {}
img_to_inference = {}
#create dictionary with concepts
for i in tqdm(concepts, total = len(concepts)):
    img_to_clue[str(i["inputs"]["image"])+str(i["inputs"]["bboxes"])] = i["knowledge"]["clue_concept"]
    img_to_inference[i["targets"]["inference"]] = i["knowledge"]["inference_concept"]
    
for i in tqdm(val, total = len(val)):
    try:    
        clu = img_to_clue[str(i["image"])+str(i["region"])]
        inf = img_to_inference[i["inference"]]
    
        i["clue_concept"] = clu
        i["inf_concept"] = inf
    except:
        continue
json.dump(val, open("/capstone/capstone-localization/val_localization/val_localization_instances_knowledge.json"))
