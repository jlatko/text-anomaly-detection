import json
import os

import numpy as np
from sacred import Experiment

import ast
from utils.corpus import get_corpus
from collections import Counter, defaultdict
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.models.callbacks import PerplexityMetric
from gensim.corpora.dictionary import Dictionary
import logging

try:
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("gensim.models.ldamodel").setLevel(logging.WARNING)
except Exception as e:
    print(e)

ex = Experiment('baseline')

np.random.seed(12)

@ex.config
def default_config():
    data_path = 'data/'
    s1 = 'friends-corpus'
    s2 = 'parliament-corpus'
    split_sentences = True
    punct = False
    to_ascii = True
    min_len = 3
    max_len = 15
    test_size = 0.1
    text_field = 'text'
    subsample_rows = None
    min_freq = 1
    num_topics = 32

@ex.capture
def train(data_path, s1, s2, split_sentences, punct, to_ascii, min_freq,
          min_len, max_len, test_size, text_field, subsample_rows, num_topics):
    # load data
    train1, val1, _, _ = get_corpus(source=s1,
                                    split_sentences=split_sentences,
                                    punct=punct,
                                    to_ascii=to_ascii,
                                    data_path=data_path,
                                    min_len=min_len,
                                    max_len=max_len,
                                    test_size=test_size,
                                    text_field=text_field,
                                    subsample_rows=subsample_rows)
    train2, val2, _, _ = get_corpus(source=s2,
                                    split_sentences=split_sentences,
                                    punct=punct,
                                    to_ascii=to_ascii,
                                    data_path=data_path,
                                    min_len=min_len,
                                    max_len=max_len,
                                    test_size=test_size,
                                    text_field=text_field,
                                    subsample_rows=subsample_rows)
    
    # build vocab  and vectorize
    t1 = train1.utterance.apply(ast.literal_eval)
    t2 = train2.utterance.apply(ast.literal_eval)
    v1_utts = val1.utterance.apply(ast.literal_eval)
    v2_utts = val2.utterance.apply(ast.literal_eval)
    d1 = Dictionary(t1)
    d2 = Dictionary(t2)
    t11 = [d1.doc2bow(t) for t in t1]
    t22 = [d2.doc2bow(t) for t in t2]

    v1_1 = [d1.doc2bow(t) for t in v1_utts]
    v1_2 = [d1.doc2bow(t) for t in v2_utts]
    v2_1 = [d2.doc2bow(t) for t in v1_utts]
    v2_2 = [d2.doc2bow(t) for t in v2_utts]


    print('LDA bound')
    print(f'{s1} vs {s2}:')
    lda = LdaMulticore(t11, num_topics=num_topics)
    # This shit is hella slow :(
    v1_scores = np.array([lda.bound([v]) for v in v1_1])
    v2_scores = np.array([lda.bound([v]) for v in v1_2])
    x = np.concatenate([v1_scores, v2_scores])
    y = np.concatenate([np.ones_like(v1_scores), np.zeros_like(v2_scores)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print('ROC AUC: ', metrics.roc_auc_score(y, x))
    
    print(f'{s2} vs {s1}:')
    lda = LdaMulticore(t22, num_topics=num_topics)
    v1_scores = np.array([lda.bound([v]) for v in v2_1])
    v2_scores = np.array([lda.bound([v]) for v in v2_2])
    x = np.concatenate([v1_scores, v2_scores])
    y = np.concatenate([np.zeros_like(v1_scores), np.ones_like(v2_scores)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print('ROC AUC: ', metrics.roc_auc_score(y, x))


@ex.automain
def main():
    train()