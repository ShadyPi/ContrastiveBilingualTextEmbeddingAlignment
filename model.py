import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class BilingualEmbeddingDataset(Dataset):
    def __init__(self, texts, model_name='distilbert/distilbert-base-multilingual-cased', device='cpu'):
        self.save_path_en = f'./embeddings/{model_name.replace("/", "-")}_en.pt'
        self.save_path_zh = f'./embeddings/{model_name.replace("/", "-")}_zh.pt'
        self.texts = texts[:131_072]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        if os.path.exists(self.save_path_en) and os.path.exists(self.save_path_zh):
            self.en_embeddings = torch.load(self.save_path_en)
            self.zh_embeddings = torch.load(self.save_path_zh)
            print(f'Load embeddings from {self.save_path_en} and {self.save_path_zh}.')
        else:
            self.en_embeddings = torch.cat([self.bert_sentence_embedding([text[0] for text in texts[i: i+64]]).cpu()
                                            for i in range(0, len(self.texts), 64)], dim=0)
            self.zh_embeddings = torch.cat([self.bert_sentence_embedding([text[1] for text in texts[i: i+64]]).cpu()
                                            for i in range(0, len(self.texts), 64)], dim=0)
            torch.save(self.en_embeddings, self.save_path_en)
            torch.save(self.zh_embeddings, self.save_path_zh)
            print(f'Save embeddings at {self.save_path_en} and {self.save_path_zh}.')

    def bert_sentence_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state.cpu()  # [batch_size, seq_length, hidden_size]
        return last_hidden_state[:, 0, :]  # [CLS] token

    def __len__(self):
        return len(self.en_embeddings)

    def __getitem__(self, idx):
        return self.en_embeddings[idx], self.zh_embeddings[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.inputFC = nn.Linear(input_dim, hidden_dim)
        self.outputFC = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inputFC(x)
        if self.num_layers > 0:
            x = self.model(x)
        x = self.outputFC(x)
        return x


class CLIPAlign(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512, num_layers=1, dropout_prob=0.5):
        super(CLIPAlign, self).__init__()
        if num_layers >= 0:
            self.en_transform = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
            self.zh_transform = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
        else:
            self.en_transform = nn.Linear(input_dim, output_dim)
            self.zh_transform = nn.Linear(input_dim, output_dim)

    def forward(self, en_embeddings, zh_embeddings):
        en_transformed = self.en_transform(en_embeddings)
        zh_transformed = self.zh_transform(zh_embeddings)
        return en_transformed, zh_transformed

    def batch_align(self, embeddings, lang, batch_size=64):
        if lang == 1:
            aligned = torch.cat([self.zh_transform(embeddings[i:i + batch_size, :])
                                 for i in range(0, embeddings.size(0), batch_size)])
        else:
            aligned = torch.cat([self.en_transform(embeddings[i:i + batch_size, :])
                                 for i in range(0, embeddings.size(0), batch_size)])
        return aligned


def nt_xent_loss(embeddings1, embeddings2, temperature=0.05):
    # L2 Normalization
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # cosine similarity
    similarity_matrix = torch.matmul(embeddings1, embeddings2.T)

    labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

    loss1 = F.cross_entropy(similarity_matrix / temperature, labels)
    loss2 = F.cross_entropy(similarity_matrix.T / temperature, labels)

    predictions = torch.argmax(similarity_matrix, dim=1)
    correct = (predictions == labels).sum().item()

    return (loss1 + loss2) / 2, correct
