import json
import os
import pickle

import numpy as np
from sacred import Experiment

import ast
from utils.corpus import get_corpus
from collections import Counter, defaultdict
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

ex = Experiment('baseline')

np.random.seed(12)

def get_log_likelihoods(vectorized, frequencies): # normalized by length
    w = vectorized.multiply(frequencies).tocsr()
    log_likes = []
    for i in range(w.shape[0]):
        _, ind = w[i].nonzero()
        if len(ind) == 0:
            print(i)
            raise Exception
        log_likes.append(sum(np.log(w[i,j]) for j in ind)/len(ind))
    return np.array(log_likes)

@ex.config
def default_config():
    data_path = 'data/'
    s1 = 'parliament-corpus'
    s2 = 'friends-corpus'
    split_sentences = True
    punct = False
    to_ascii = True
    min_len = 3
    max_len = 15
    test_size = 0.1
    text_field = 'text'
    subsample_rows = None
    min_freq = 1

@ex.capture
def train(data_path, s1, s2, split_sentences, punct, to_ascii, min_freq,
          min_len, max_len, test_size, text_field, subsample_rows):
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
    
    # build vocab
    print('Building vocab')
    total_vocab = list(set(w for df in [train1, val1, train2, val2] for utt in df.utterance.apply(ast.literal_eval) for w in utt))

    # train frequencies
    print('Creating vectorizers')
    eps = 1e-8
    vect = CountVectorizer(ngram_range=(1,1), analyzer='word', vocabulary=total_vocab, 
                            token_pattern = r"(?u)\b\w+\b", min_df=min_freq)
    v1 = vect.transform(train1.utterance.apply(ast.literal_eval).apply(' '.join))
    v2 = vect.transform(train2.utterance.apply(ast.literal_eval).apply(' '.join))
    freq1 = v1.sum(axis=0)[0]/v1.sum() + eps
    freq2 = v2.sum(axis=0)[0]/v2.sum() + eps

    # validation data
    v1_val = vect.transform(val1.utterance.apply(ast.literal_eval).apply(' '.join))
    v2_val = vect.transform(val2.utterance.apply(ast.literal_eval).apply(' '.join))

    # general frequencies
    # https://www.kaggle.com/rtatman/english-word-frequency
    print('Getting general frequencies')
    general_word_freqs = pd.read_csv('data/unigram_freq.csv')
    freq_dict = defaultdict(int, general_word_freqs.set_index('word')['count'].to_dict())
    general_freq = np.array([freq_dict[w] for w in total_vocab])
    general_freq = general_freq/sum(general_freq) + eps

    # calculate likelihoods
    print('Calculating likelihoods')
    ll1_f1 = get_log_likelihoods(v1_val, freq1) # likelihoods of samples in 1 given frequencies in 1
    ll2_f1 = get_log_likelihoods(v2_val, freq1) # likelihoods of samples in 2 given frequencies in 1
    ll1_f2 = get_log_likelihoods(v1_val, freq2) 
    ll2_f2 = get_log_likelihoods(v2_val, freq2) 
    ll1_general = get_log_likelihoods(v1_val, general_freq) 
    ll2_general = get_log_likelihoods(v2_val, general_freq)

    with open(f'{data_path}/general_likelihoods_{s1}.pickle', 'wb') as fh:
        pickle.dump(ll1_general, fh)
    with open(f'{data_path}/general_likelihoods_{s2}.pickle', 'wb') as fh:
        pickle.dump(ll2_general, fh)

    print('Likelihood approach (normalized by length)')
    x = np.concatenate([ll1_f1, ll2_f1])
    y = np.concatenate([np.ones_like(ll1_f1), np.zeros_like(ll2_f1)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print(f'{s1} vs {s2}:')
    print('ROC AUC: ', metrics.roc_auc_score(y, x))

    x = np.concatenate([ll2_f2, ll1_f2])
    y = np.concatenate([np.ones_like(ll2_f2), np.zeros_like(ll1_f2)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print(f'{s2} vs {s1}:')
    print('ROC AUC: ', metrics.roc_auc_score(y, x))


    print('Likelihood ratio')
    x = np.concatenate([ll1_f1 - ll1_general, ll2_f1 - ll2_general])
    y = np.concatenate([np.ones_like(ll1_f1), np.zeros_like(ll2_f1)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print(f'{s1} vs {s2}:')
    print('ROC AUC: ', metrics.roc_auc_score(y, x))

    x = np.concatenate([ll2_f2 - ll2_general, ll1_f2 - ll1_general])
    y = np.concatenate([np.ones_like(ll2_f2), np.zeros_like(ll1_f2)]).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print(f'{s2} vs {s1}:')
    print('ROC AUC: ', metrics.roc_auc_score(y, x))

@ex.automain
def main():
    train()