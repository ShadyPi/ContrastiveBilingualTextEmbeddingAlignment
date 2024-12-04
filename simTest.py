import numpy as np
import pandas as pd
import torch.cuda
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from matplotlib import rcParams
from model import CLIPAlign

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def bert_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state.cpu()  # [batch_size, seq_length, hidden_size]
    return last_hidden_state[:, 0, :].numpy()  # [CLS] token


def cosine_similarity_matrix(vectors1, vectors2):
    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    normalized_vectors1 = vectors1 / norms1
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
    normalized_vectors2 = vectors2 / norms2

    similarity_matrix = np.dot(normalized_vectors1, normalized_vectors2.T)

    return similarity_matrix


def draw(data, title, xticks=None, yticks=None):
    # np.fill_diagonal(data, np.nan)
    plt.figure(figsize=(8, 8))
    # plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.imshow(data, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

    plt.colorbar()

    n = len(data)
    if xticks is None:
        plt.xticks(ticks=np.arange(0, n, 1), labels=np.arange(1, n + 1, 1))
        plt.yticks(ticks=np.arange(0, n, 1), labels=np.arange(1, n + 1, 1))
    else:
        plt.xticks(ticks=np.arange(0, n, 1), labels=xticks, fontsize=20)
        plt.yticks(ticks=np.arange(0, n, 1), labels=yticks, fontsize=20, rotation='vertical')
    plt.xlabel("Chinese Texts", fontsize=25)
    plt.ylabel("English Texts", fontsize=25)
    # plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

    # plt.title(title)
    plt.savefig(f'./pics/{title}.png', bbox_inches='tight')


def STEmb(model_name, en_text, zh_text):
    model = SentenceTransformer(model_name)
    en_embeddings = model.encode(en_text, convert_to_numpy=True)
    zh_embeddings = model.encode(zh_text, convert_to_numpy=True)
    return en_embeddings, zh_embeddings


def BERTEmb(model_name, en_text, zh_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    en_embeddings = np.concatenate([bert_sentence_embedding(en_text, tokenizer, model)], axis=0)
    zh_embeddings = np.concatenate([bert_sentence_embedding(zh_text, tokenizer, model)], axis=0)
    return en_embeddings, zh_embeddings


def AlignedEmb(model_name, aligner_path, en_text, zh_text):
    aligner = CLIPAlign(input_dim=768, hidden_dim=1024, output_dim=768, num_layers=-1)
    aligner.load_state_dict(torch.load(aligner_path))
    aligner = aligner.to(device)
    aligner.eval()
    en_embeddings, zh_embeddings = BERTEmb(model_name, en_text, zh_text)
    en_embeddings, zh_embeddings = torch.from_numpy(en_embeddings).to(device), torch.from_numpy(zh_embeddings).to(device)
    en_embeddings, zh_embeddings = aligner(en_embeddings, zh_embeddings)
    en_embeddings, zh_embeddings = en_embeddings.detach().cpu().numpy(), zh_embeddings.detach().cpu().numpy()
    return en_embeddings, zh_embeddings


df = pd.read_csv('./data/bioClustering/bio_clustering_samples.csv')
en_texts = df['sentences'].tolist()
zh_texts = df['sentences_zh'].tolist()

en_embeddings, zh_embeddings = STEmb('distiluse-base-multilingual-cased-v1', en_texts, zh_texts)
sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
draw(sim, 'SentenceBERT Similarity')

en_embeddings, zh_embeddings = BERTEmb('distilbert/distilbert-base-multilingual-cased', en_texts, zh_texts)
sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
draw(sim, 'DistilBERT Similarity')

en_embeddings, zh_embeddings = AlignedEmb('distilbert/distilbert-base-multilingual-cased',
                                          './models/aligner_-1_768_98.24.pt', en_texts, zh_texts)
sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
draw(sim, 'Aligned Similarity -1 768 9605')

# en_texts = ['Hello', 'Goodbye', 'Cat', 'Dog']
# zh_texts = ['你好', '再见', '猫', '狗']

# en_embeddings, zh_embeddings = STEmb('distiluse-base-multilingual-cased-v1', en_texts, zh_texts)
# sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
# draw(sim, 'SentenceBERT Sample', zh_texts, en_texts)
#
# en_embeddings, zh_embeddings = BERTEmb('distilbert/distilbert-base-multilingual-cased', en_texts, zh_texts)
# sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
# draw(sim, 'Multilingual DistilBERT Sample Auto', zh_texts, en_texts)
#
# en_embeddings, zh_embeddings = BERTEmb('distilbert/distilbert-base-cased', en_texts, zh_texts)
# sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
# draw(sim, 'Base DistilBERT Sample Auto', zh_texts, en_texts)

# en_embeddings, zh_embeddings = AlignedEmb('distilbert/distilbert-base-multilingual-cased',
#                                           './models/aligner_-1_ 98.21.pt', en_texts, zh_texts)
# sim = cosine_similarity_matrix(en_embeddings, zh_embeddings)
# draw(sim, 'Aligned Sample Auto')
