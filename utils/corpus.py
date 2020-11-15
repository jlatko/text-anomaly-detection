from convokit import Corpus, download
from paths import DATA_DIR
import os
from cleantext import clean
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from utils.preprocess import flatten_sentences
import numpy as np
from sklearn.model_selection import train_test_split

nltk.download('punkt')
punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~' # no '

def get_corpus(source, split_sentences=False, punct=True, to_ascii=True,
               min_len=3, max_len=15, test_size=0.1, text_field='text', subsample_rows=False, save=True):
    if source.endswith('.csv'):
        csv = True
        name = source[:-4]
    else:
        csv = False
        name = source 

    # compose name
    corpus_name = f'{name}{"_split" if split_sentences else ""}' \
                  f'{"_punct" if punct else ""}' \
                  f'{"_ascii" if to_ascii else ""}' \
                  f'{f"_sub{subsample_rows}" if subsample_rows else ""}' \
                  f'_{test_size}_{min_len}_{max_len}' 
    corpus_train = os.path.join(DATA_DIR, f'{corpus_name}_train.csv')
    corpus_test = os.path.join(DATA_DIR, f'{corpus_name}_test.csv')

    # Load from cache
    if os.path.isfile(corpus_train) and os.path.isfile(corpus_test):
        df_train, df_val = pd.read_csv(corpus_train), pd.read_csv(corpus_test)
        print('Loading cached data...')
        print(len(df_train))
        print(len(df_val))
        return df_train, df_val, f'{corpus_name}_train.csv', f'{corpus_name}_test.csv'
    
    # load csv or download
    if csv:
        print('Loading dataset from csv...')
        df = pd.read_csv(os.path.join(DATA_DIR, source))
    else:
        print('Downloading dataset...')
        corp = Corpus(filename=download(name))
        df = corp.get_utterances_dataframe()

    # get only text
    df = df.rename(columns={text_field: "utterance"})[["utterance"]]
    
    # remove any tags
    df['utterance'] = df['utterance'].str.replace(r'<.*>', ' ') 

    # subsample
    if subsample_rows:
        df = df.sample(subsample_rows, random_state=0)

    # split sentences
    if split_sentences:
        sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        df["utterance"] = df["utterance"]
        df["utterance"] = df["utterance"].apply(sentence_detector.tokenize)
        df = flatten_sentences(df)
    
    print('Cleaning')
    cln_fn = lambda x: clean(x,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=to_ascii,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        lang="en"                       # set to 'de' for German special handling
    )

    # clean
    df["utterance"] = df["utterance"].apply(cln_fn)
    if not punct:
        df["utterance"] = df["utterance"].str.replace(r"[{}]".format(punctuation),' ')

    # tokenize
    sen_by_words = df["utterance"].apply(word_tokenize)
    word_counts = sen_by_words.apply(len)
    sen_by_words = sen_by_words[(word_counts <= max_len) & (word_counts >= min_len)]
    df = sen_by_words.to_frame()

    # split
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=0)
    print(len(df_train))
    print(len(df_val))
    if not save:
        return df_train, df_val
    df_train.to_csv(corpus_train, index=False)
    df_val.to_csv(corpus_test, index=False)
    return df_train, df_val, f'{corpus_name}_train.csv', f'{corpus_name}_test.csv'
