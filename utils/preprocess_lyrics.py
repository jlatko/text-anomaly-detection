import pandas as pd
from paths import DATA_DIR
from nltk.tokenize import word_tokenize


def preprocess_lyrics(df, lower_word_count=3, upper_word_count=15):
    df["utterance"] = df["utterance"].str.lower()
    df["utterance"] = df["utterance"].apply(word_tokenize)
    word_counts = df["utterance"].apply(len)

    df["utterance"] = df["utterance"][(word_counts <= upper_word_count) & (word_counts >= lower_word_count)]
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df1 = pd.read_csv(DATA_DIR / "archive" / "amy-winehouse.txt", header=None, names=["utterance"])
    df2 = pd.read_csv(DATA_DIR / "archive" / "bjork.txt", header=None, names=["utterance"])

    df1 = preprocess_lyrics(df1)
    df2 = preprocess_lyrics(df2)

    df1.to_csv(DATA_DIR / "Amy.csv", index=False)
    df2.to_csv(DATA_DIR / "Bjork.csv", index=False)
