import json
import sys, os
from sentence_transformers import SentenceTransformer
import pickle

with open(sys.argv[1], 'r') as f:
    sentences = [json.loads(item)['input'] for item in f.readlines()]

model = SentenceTransformer(sys.argv[2], device='cuda')
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences, batch_size=32)

#Store sentences & embeddings on disc
if not os.path.exists(f'./output/{sys.argv[3]}'):
    os.makedirs(f'./output/{sys.argv[3]}')
    
with open(f'./output/{sys.argv[3]}/embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

