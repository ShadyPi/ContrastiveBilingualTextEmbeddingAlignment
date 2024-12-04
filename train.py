import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from model import *
from torch.utils.data import DataLoader, random_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
num_epochs = 50
batch_size = 64
lr = 1e-1
wd = 5e-4
loss_temp = 0.05
num_layers = 2
output_dim = 768


def valid(aligner, dataloader):
    aligner.eval()
    loss_sum = 0
    correct_sum = 0
    for en_embed, zh_embed in dataloader:
        en_embed, zh_embed = en_embed.to(device), zh_embed.to(device)
        en_transformed, zh_transformed = aligner(en_embed, zh_embed)
        loss, correct = nt_xent_loss(en_transformed, zh_transformed, temperature=loss_temp)

        loss_sum += loss.item()
        correct_sum += correct
    return loss_sum, correct_sum


def test(timestamp, aligner, en_embeds, zh_embeds, labels, num_clusters):
    aligner.eval()
    en_transformed = aligner.batch_align(en_embeds, 0)
    zh_transformed = aligner.batch_align(zh_embeds, 1)
    en_transformed = F.normalize(en_transformed, p=2, dim=1)
    zh_transformed = F.normalize(zh_transformed, p=2, dim=1)
    embeddings = torch.cat([en_transformed, zh_transformed], dim=0)
    embeddings = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
    predicted_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, predicted_labels)
    if timestamp:
        pred = pd.DataFrame({'pred_labels': predicted_labels})
        pred.to_csv(f'./results/test_results_{num_layers}_{timestamp}.csv', index=False)
    return ari


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    timestamp = time.time()
    file_handler = logging.FileHandler(f'logs/train_{num_layers}_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f'seed {seed}, num_epochs {num_epochs}, batch_size {batch_size},'
                f' lr {lr}, wd {wd}, loss_temp {loss_temp}, num_layers {num_layers}, output_dim {output_dim}')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Train & Valid Data
    with open('./data/WMT18/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = BilingualEmbeddingDataset(data, model_name='distilbert/distilbert-base-multilingual-cased', device=device)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    print(len(train_dataset), len(val_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Test Data
    df = pd.read_csv('data/bioClustering/bio_clustering_enzh_42.csv')
    num_clusters = len(df['labels'].unique())
    sentences = df['sentences'].tolist()
    labels = df['labels'].tolist()
    langs = df['lang'].tolist()
    embeddings = torch.cat([dataset.bert_sentence_embedding(sentences[i: i + 64])
                              for i in range(0, len(sentences), 64)], dim=0)
    label_map = [[], []]
    en_embeds, zh_embeds = [], []
    for i in range(len(langs)):
        label_map[langs[i]].append(labels[i])
        if langs[i] == 0:
            en_embeds.append(embeddings[i, :])
        else:
            zh_embeds.append(embeddings[i, :])
    en_embeds = torch.stack(en_embeds, dim=0)
    zh_embeds = torch.stack(zh_embeds, dim=0)
    en_embeds, zh_embeds = en_embeds.to(device), zh_embeds.to(device)
    labels = label_map[0] + label_map[1]

    # Model Training
    aligner = CLIPAlign(input_dim=dataset.en_embeddings.size(-1), hidden_dim=1024, output_dim=output_dim, num_layers=num_layers)
    aligner = aligner.to(device)
    optimizer = torch.optim.AdamW(aligner.parameters(), lr=lr, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # embeddings = torch.cat([en_embeds, zh_embeds], dim=0)
    # embeddings = embeddings.detach().cpu().numpy()
    # kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
    # predicted_labels = kmeans.fit_predict(embeddings)
    # init_ari = adjusted_rand_score(labels, predicted_labels)
    # print(f'Initial ARI: {init_ari}')
    # best_ari = 0
    start = time.perf_counter()
    best_val_acc = 0
    for epoch in range(num_epochs):
        loss_sum = 0
        correct_sum = 0
        aligner.train()
        for en_embed, zh_embed in train_dataloader:
            en_embed, zh_embed = en_embed.to(device), zh_embed.to(device)
            en_transformed, zh_transformed = aligner(en_embed, zh_embed)
            loss, correct = nt_xent_loss(en_transformed, zh_transformed, temperature=loss_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct_sum += correct
        val_loss, val_correct = valid(aligner, val_dataloader)
        val_loss /= len(val_dataset)
        val_correct /= len(val_dataset)
        if val_correct > best_val_acc and val_correct > 0.8:
            best_val_acc = val_correct
            torch.save(aligner.state_dict(), f'./models/aligner_{num_layers}_{output_dim}_{val_correct*100.0:.2f}.pt')
        test_ari = test(None, aligner, en_embeds, zh_embeds, labels, num_clusters)
        scheduler.step(val_loss)
        logger.info(f'### Epoch {epoch}')
        logger.info(f"Train Loss: {loss_sum/len(train_dataset)}, Train Acc: {correct_sum/len(train_dataset)}")
        logger.info(f"Valid Loss: {val_loss}, Valid Acc: {val_correct}")
        logger.info(f'Test ARI: {test_ari}')
        # best_ari = max(best_ari, test_ari)
    end = time.perf_counter()
    run_time = end - start
    aligner.load_state_dict(torch.load(f'./models/aligner_{num_layers}_{output_dim}_{best_val_acc*100.0:.2f}.pt'))
    test_ari = test(timestamp, aligner, en_embeds, zh_embeds, labels, num_clusters)
    logger.info(f'Best Val Acc: {best_val_acc}')
    logger.info(f'Test ARI of Best Val Acc Model: {test_ari}')
    logger.info(f'Run Time: {run_time}')


if __name__ == '__main__':
    main()
