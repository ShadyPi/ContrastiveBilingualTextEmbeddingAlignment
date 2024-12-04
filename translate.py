import numpy as np
import pandas as pd
from googletrans import Translator
from tqdm import tqdm


def translate(begin, end):
    translator = Translator()
    print(translator.translate('hello', src='en', dest='zh-cn').text)

    df = pd.read_json("hf://datasets/mteb/biorxiv-clustering-s2s/test.jsonl.gz", lines=True)
    df = df.head(end).tail(end-begin)
    print(len(df))
    trans = []

    for index, row in tqdm(df.iterrows()):
        trans.append(translator.translate(row['sentences'], src='en', dest='zh-cn').text)
    df['sentences_zh'] = trans

    df.to_csv(f'./data/bioClustering/bio_clustering_{begin}_{end}.csv', index=False)


def merge(begin, end, step):
    pd_list = []
    for b in range(begin, end, step):
        e = b+step
        new_pd = pd.read_csv(f'./data/bioClustering/bio_clustering_{b}_{e}.csv')
        pd_list.append(new_pd)
    full_pd = pd.concat(pd_list)
    full_pd.to_csv(f'./data/bioClustering/bio_clustering.csv', index=False)
    return full_pd


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    # for b in range(0, 1000, 200):
    #     try:
    #         translate(b, b+200)
    #     except Exception as e:
    #         print(e)
    #         translate(b, b+200)

    fullData = merge(0, 1000, 200)
    choice = np.random.rand(len(fullData))
    choice = (choice > 0.5).astype(int)
    chosenSentences = []
    for i, row in fullData.iterrows():
        if choice[i] == 0:
            chosenSentences.append(row['sentences'])
        else:
            chosenSentences.append(row['sentences_zh'])

    testData = pd.DataFrame({'sentences': chosenSentences, 'labels': fullData['labels'], 'lang': choice})
    # testData['labels'] = fullData['labels']
    testData.to_csv(f'./data/bioClustering/bio_clustering_enzh_{seed}.csv', index=False)


