import json
import sys, os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

with open('../data_preprocessing/mimic_data_for_retrieve.json', 'r') as f:
    data = json.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

for key in enumerate(data):

    corpus = data[key]['sentences']
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, batch_size=16)
    queries = data[key]['questions']

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))
    results = {}
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        results[query] = []

        for score, idx in zip(top_results[0], top_results[1]):
            results[query].append([corpus[idx], "Score:{:.4f}".format(score)])
            
    data[key]['results'] = results


with open('./mimic_data_for_retrieve_sentence_bert_result.json', 'w') as f:
    json.dump(data, f, indent=2)