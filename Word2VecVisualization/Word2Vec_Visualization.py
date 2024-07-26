#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gensim
import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


BASE_PATH = os.getcwd()
CORPUS = "spam_email.txt"


CORPORA_PATH = os.path.join(BASE_PATH, "corpora", CORPUS)
PLOTS_PATH = os.path.join(BASE_PATH, "plots")
MODELS_PATH = os.path.join(BASE_PATH, "models")
TOKENS_TO_PLOT_COUNT = 500


class SentenceIterator:
    def __init__(self, filename=""):
        self.filename = filename

    def zzz(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding="utf-8"):
                yield line.split() # generator

    def __iter__(self):
        for line in open(self.filename, mode="r", encoding="utf-8"):
            yield line.split()  # generator


def plot_with_labels(low_dim_embs, labels, filename=''):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    print(os.path.abspath(filename))


def compute_tsne(model=None):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', max_iter=5000, method='exact')

    # Get word vectors for each word.
    # https://stackoverflow.com/questions/40581010/how-to-run-tsne-on-word2vec-created-from-gensim
    # final_embeddings = model[model.wv.vocab]  # gensim 3.8.3
    word_embeddings = model.wv.vectors
    # >>> model.wv.vectors.shape
    # (5378, 128)
    # >>> len(model.wv.index_to_key)
    # 5378
    # >>> model.wv.index_to_key[:5]
    # ['view', 'shop', 'com', 'shipping', '\u200c']

    lower_dimensional_embeddings = tsne.fit_transform(word_embeddings[:TOKENS_TO_PLOT_COUNT, :])
    labels = list(model.wv.index_to_key)[0: TOKENS_TO_PLOT_COUNT]
    return lower_dimensional_embeddings, labels


def plot_tokens(embeddings, labels):
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    fig, ax = plt.subplots()
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
    plt.savefig('tsne.png')


def save_model(model):
    try:
        print('Begin SAVE MODEL')
    except Exception as ex:
        print(ex)


def run_inferences(model):
    try:
        print(model.wv['Picard'])
        print(model.wv.similarity('Klingon', 'Romulan'))
        print(model.wv.most_similar(positive=['Picard']))
    except Exception as e:
        print(e)


def main():
    with open(CORPORA_PATH, mode="r", encoding="utf-8") as f:
        print("FIRST FIVE PRE-PROCESSED SENTENCES")
        for i in range(5):
            line = f.readline()[:-1]
            print(line)  # Truncate the trailing newline.
        print()

    sentences = SentenceIterator(filename=CORPORA_PATH)
    model = gensim.models.Word2Vec(sentences, sorted_vocab=1, sg=1, min_count=5, epochs=15, vector_size=128,
                                   batch_words=128, window=5, compute_loss=False, alpha=0.2)
    print(f"Model Training Loss: {model.get_latest_training_loss()}")
    print()

    print(f"{'Order':>4} {'Token':>15} {'Count':>8}")
    for idx, token in enumerate(model.wv.index_to_key):
        token_to_print = token
        if token is None:
            token_to_print = ''
        if len(token) == 1 and ord(token) == 8204:  # 0x200c ZERO-WIDTH NON-JOINER
            token_to_print = hex(ord(token))
        print(f"{idx:>4} {token_to_print:>15} {model.wv.get_vecattr(token, "count"):>8}")
        if idx > 9:
            break
    print()

    model.build_vocab()
    model.save(os.path.join(MODELS_PATH, "spam_email.mdl"))

    # Setup plotting
    date_time_string = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    lower_dimensional_embeddings, labels = compute_tsne(model=model)
    plot_with_labels(lower_dimensional_embeddings, labels)


if __name__ == '__main__':
    main()
