# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:31:46 2019

@author: PEECO
"""

# Latent Semantic Analysis

# import the necessary modules
from sklearn.datasets import fetch_20newsgroups
from gensim import corpora
from gensim.models import LsiModel
from gensim.models import LdaModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# fetch newsgroup data based on the listed categories only
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
document_list = twenty_train.data
document_title = []

# Preprocessing data
# append document tttles to list
for t in twenty_train.target:
    document_title.append(twenty_train.target_names[t])

# use regex tokenizer '\W+' for one or more characters inluding digits while 'r' is for raw/;iteral string to ignore escape chars  
tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
texts = []

# iterate through the docs to remove stopwords and stem the rest
for doc in document_list:
    raw_docs = doc.lower()
    tokens = tokenizer.tokenize(raw_docs)
    stop_stem_tokens = [ps.stem(i) for i in tokens if i not in set(stopwords.words('english'))]
    texts.append(stop_stem_tokens)
    
# prepare corpus  
# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(texts)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]

# determining the number of topics using coherence scores
def compute_coherence_values(dic, dtm, txt, stop, start):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop):
        model = LsiModel(dtm, num_topics= num_topics, id2word=dic) #train model
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts = txt, dictionary=dic, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    x = range(start, stop)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return
    #return model_list, coherence_values

#def plot_graph(dic, dtm, txt, start, stop):
#    model_list, coherence_values = compute_coherence_values(dic, dtm, stop, start)
#    x = range(start, stop)
#    plt.plot(x, coherence_values)
#    plt.xlabel("Number of Topics")
#    plt.ylabel("Coherence score")
#    plt.legend(("coherence_values"), loc='best')
#    plt.show()
#    return
    

# creating a Latent Semantic Analysis model using gensim
lsa_model = LSiModel(doc_term_matrix, num_topics=, id2word=dictionary)
print(lsa_model.print_topics(num_topics, num_words=))
