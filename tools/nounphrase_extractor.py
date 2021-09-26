import spacy
import json
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

# dataset_dir = '/home/hoang/Datasets/MIMIC/'
dataset_dir = '/home/hoang/Datasets/NLMCXR/'
count_sentence = json.load(open(dataset_dir + 'count_sentence.json', 'r'))

np_count = {}
for k,v in tqdm(count_sentence.items()):
    doc = nlp(k)
    for np in doc.noun_chunks:
        if np.text not in np_count:
            np_count[np.text] = v
        else:
            np_count[np.text] += v

json.dump(np_count, open(dataset_dir + 'count_nounphrase.json', 'w'))
            

            
    
