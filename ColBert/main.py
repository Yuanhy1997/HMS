from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import colbert

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300 # truncate passages at 300 tokens
max_id = 10000

index_name = f'Mimic.{nbits}bits'
checkpoint = 'colbert-ir/colbertv2.0'

with open('../data_preprocessing/mimic_data_for_retrieve.json', 'r') as f:
    data = json.load(f)

for key in tqdm(data):

    collection = data[key]['sentences']
    queries = data[key]['questions']

    with Run().context(RunConfig(nranks=1,rank=1, experiment='notebook')):  # nranks specifies the number of GPUs to use

        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    with Run().context(RunConfig(experiment='notebook')):

        searcher = Searcher(index=index_name, collection=collection)

    qas = {}
    for query in queries:
        qas[query] = []
        results = searcher.search(query, k=5)
        for passage_id, passage_rank, passage_score in zip(*results):
            qas[query].append([searcher.collection[passage_id], "Score:{:.4f}".format(passage_score)])

    data[key]['results'] = qas


with open('./mimic_data_for_retrieve_colbert_result.json', 'w') as f:
    json.dump(data, f, indent=2)