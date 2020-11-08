import os

import pandas as pd
import nltk.data
from nltk.tokenize import word_tokenize
import numpy as np

from paths import DATA_DIR

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def preprocess_IMDB_sentence(sentence):
    # TODO consider if we should leave non word characters

    sentence = sentence \
        .replace("<br />", " ") \
        .lower()

    words = tokenizer.tokenize(sentence)
    processed_words = word_tokenize(words)
    return processed_words


def preprocess_IMDB(df, lower_word_count=3, upper_word_count=15):
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    df["utterance"] = df["utterance"] \
        .str.replace("<br />", " ") \
        .str.lower()
    df["utterance"] = df["utterance"].apply(sentence_detector.tokenize)
    df_sent = flatten_sentences(df)
    sen_by_words = df_sent["utterance"].apply(word_tokenize)
    
    word_counts = sen_by_words.apply(len)
    sen_by_words = sen_by_words[(word_counts <= upper_word_count) & (word_counts >= lower_word_count)]
    return sen_by_words


# TODO check how \' works with embeddings

def flatten_sentences(df):
    return df["utterance"].apply(lambda x: pd.Series(x)) \
        .stack() \
        .reset_index(drop=True) \
        .to_frame("utterance")


if __name__ == "__main__":
    # https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv
    # 1. Download
    # 2. unzip
    # 3. move to DATA_DIR
    # 4. set path_source to csv name file
    # 5. run this script

    path_source = os.path.join(DATA_DIR, "IMDB Dataset.csv")
    path_target_train = os.path.join(DATA_DIR, "traindf.csv")
    path_target_valid = os.path.join(DATA_DIR, "valdf.csv")
    path_target_whole = os.path.join(DATA_DIR,
                                     "wholedf.csv")  # If you want to change split without processing everything

    df = pd.read_csv(path_source, usecols=["review"])
    df = df.rename(columns={"review": "utterance"})
    df = preprocess_IMDB(df, )

    split_ratio = 0.95

    mask = np.random.rand(len(df)) < split_ratio
    df_train = df[mask]
    df_val = df[~mask]

    print(len(df_train))
    print(len(df_val))

    df_train.to_csv(path_target_train, index=False)
    df_val.to_csv(path_target_valid, index=False)
    df.to_csv(path_target_whole, index=False)
