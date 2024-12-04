import argparse

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from transformers import AutoTokenizer, AutoModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def bert_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state.cpu()  # [batch_size, seq_length, hidden_size]
    return last_hidden_state[:, 0, :].numpy()  # [CLS] token


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Embedding Method')
    parser.add_argument('model', type=str, help='Embedding Model')
    parser.add_argument('dataset', type=str, help='Dataset for evaluation')

    args = parser.parse_args()
    print(args.mode, args.model)

    if args.dataset == 'enzh':
        df = pd.read_csv('data/bioClustering/bio_clustering_enzh_42.csv')
        sentences = df['sentences'].tolist()
    elif args.dataset == 'zh':
        df = pd.read_csv('data/bioClustering/bio_clustering.csv')
        sentences = df['sentences_zh'].tolist()
    else:
        df = pd.read_csv('data/bioClustering/bio_clustering.csv')
        sentences = df['sentences'].tolist()
    # df = pd.read_json("hf://datasets/mteb/biorxiv-clustering-s2s/test.jsonl.gz", lines=True)
    # df = df.head(1000)
    # sentences = df['sentences'].tolist()
    labels = df['labels'].tolist()
    num_clusters = len(df['labels'].unique())
    print(len(df), num_clusters)

    if 'CLS' in args.mode:
        # python baseline.py CLS distilbert/distilroberta-base
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        model = model.to(device)
        model.eval()
        embeddings = np.concatenate([bert_sentence_embedding(sentences[i: i+64], tokenizer, model)
                                     for i in range(0, len(sentences), 64)], axis=0)
    else:
        # python baseline.py ST all-distilroberta-v1
        # python baseline.py ST distiluse-base-multilingual-cased-v1
        assert 'ST' in args.mode, 'mode not defined'
        model = SentenceTransformer(args.model)
        embeddings = model.encode(sentences, convert_to_numpy=True)

    print(embeddings.shape)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
    predicted_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari}")

