# based on: https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
from torchtext import data, vocab
import ast
from datasets.batch_generator import BatchGenerator
from hyperparams import BATCH_SIZE, WORD_EMBEDDING_SIZE
from paths import DATA_DIR, GLOVE_DIR


def init_data_loading():
    utterance_field = data.Field(sequential=True,
                                 tokenize=lambda s: ast.literal_eval(s),
                                 fix_length=15,
                                 batch_first=True,
                                 use_vocab=True,
                                 is_target=True,
                                 unk_token='<unk>',
                                 pad_token='<pad>')

    train_val_fields = [
        ('utterance', utterance_field)
    ]

    train_dataset, val_dataset = data.TabularDataset.splits(path=DATA_DIR,
                                                            format='csv',
                                                            train='traindf.csv',
                                                            validation='valdf.csv',
                                                            fields=train_val_fields,
                                                            skip_header=True)

    # specify the path to the localy saved vectors
    vec = vocab.Vectors(f'glove.6B.{WORD_EMBEDDING_SIZE}d.txt',
                        GLOVE_DIR)  # download here: https://nlp.stanford.edu/projects/glove/

    utterance_field.build_vocab(train_dataset, val_dataset, vectors=vec)

    print(utterance_field.vocab.vectors[utterance_field.vocab.stoi['the']])

    train_loader, val_loader = data.BucketIterator.splits(datasets=(train_dataset, val_dataset),
                                                          batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                          repeat=False)

    batch = next(iter(train_loader))  # BucketIterator return a batch object
    train_batch_it = BatchGenerator(train_loader, 'utterance')
    # print(next(iter(train_batch_it)))
    datasets = (train_dataset, val_dataset)
    data_loaders = (train_loader, val_loader)
    return datasets, data_loaders, utterance_field

