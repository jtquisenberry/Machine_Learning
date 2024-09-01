#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install dependencies
get_ipython().system('pip install "scipy<1.13"')
get_ipython().system('pip install numpy')
get_ipython().system('pip install gensim')


# In[4]:


import numpy
import scipy
import gensim


# In[5]:


CORPUS_PATH = r"D:\W\Development\Machine_Learning\corpora\spam_email\spam_email_sentences.txt"


# In[10]:


with open(CORPUS_PATH, mode="r", encoding="utf-8") as f:
    for i in range(3):
        print(f.readline()[:-1])  # Truncate the trailing newline.


# In[17]:


class SentenceIterator:
    def __init__(self, filename=""):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, mode="r", encoding="utf-8"):
            yield line.split()  # generator


# In[18]:


sentences = SentenceIterator(filename=CORPUS_PATH)


# In[28]:


s2 = list(sentences)


# In[29]:


s2[:5]


# In[26]:


model = gensim.models.Word2Vec(sentences, sorted_vocab=1, sg=1, min_count=5, epochs=15, vector_size=128,
                               batch_words=128, window=5, compute_loss=False, alpha=0.2)
print(model.get_latest_training_loss())


# In[42]:


for idx, token in enumerate(model.wv.index_to_key):
    if token is None:
        print(f"{idx:>4} {'':>16} {model.wv.get_vecattr(token, "count"):>8}")
    else:
        print(f"{idx:>4} {token:>15} {model.wv.get_vecattr(token, "count"):>8}")
    if idx > 9:
        break


# In[ ]:




