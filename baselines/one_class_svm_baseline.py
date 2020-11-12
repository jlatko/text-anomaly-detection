import ast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import hyperopt
from torchtext import vocab

from paths import DATA_DIR, GLOVE_DIR


class OneClassSVM_Baseline:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.pipe = None

    def fit(self, X):
        steps = [
            ('reduce_dim', PCA(n_components=self.n_components)),
            ('svm', OneClassSVM(kernel="rbf", gamma="scale"))]

        self.pipe = Pipeline(steps)
        self.pipe.fit(X)

    def eval(self, Y, y_true):
        y_hat = self.pipe.predict(Y)

        auc = roc_auc_score(y_true, y_hat)
        acc = accuracy_score(y_true, y_hat)
        f1 = f1_score(y_true, y_hat)

        return {'roc_auc_score:': auc,
                'accuracy': acc,
                'F1': f1}


if __name__ == '__main__':
    # print(utterance_field.vocab.vectors[utterance_field.vocab.stoi['the']])
    TEST_SIZE = 500
    kanye_df = pd.read_csv(DATA_DIR / "Kanye.csv")
    kanye_df.utterance = kanye_df.utterance.apply(lambda s: ast.literal_eval(s))
    cohen_df = pd.read_csv(DATA_DIR / "Amy.csv", nrows=TEST_SIZE)
    cohen_df.utterance = cohen_df.utterance.apply(lambda s: ast.literal_eval(s))

    vec = vocab.Vectors(f'glove.6B.{300}d.txt', GLOVE_DIR)


    def to_word_vecs(words):
        return np.mean([vec.vectors[vec.stoi.get(w, np.random.randint(0, len(vec.stoi)))].numpy() for w in words],
                       axis=0)


    kanye_df["embedding"] = kanye_df["utterance"].apply(to_word_vecs)
    cohen_df["embedding"] = cohen_df["utterance"].apply(to_word_vecs)

    oc_svm = OneClassSVM_Baseline()
    ye = np.array([kanye_df["embedding"][i] for i in range(len(kanye_df["embedding"]))])
    X = ye[0:-TEST_SIZE]
    oc_svm.fit(X)
    Y = np.array([cohen_df["embedding"][i] for i in range(len(cohen_df["embedding"]))])
    Y = np.vstack([Y[:TEST_SIZE], ye[-TEST_SIZE:]])
    y_true = np.ones(TEST_SIZE * 2)
    y_true[:TEST_SIZE] = y_true[:TEST_SIZE] * -1
    print(oc_svm.eval(Y, y_true))

# {'roc_auc_score:': 0.62, 'accuracy': 0.62, 'F1': 0.56}
