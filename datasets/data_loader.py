# based on: https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
from torchtext import data, vocab
from torchtext.vocab import GloVe

import ast
from datasets.batch_generator import BatchGenerator
from paths import DATA_DIR


def init_data_loading(data_path, train_batch_size, val_batch_size, emb_size,
                      train_source, val_source, min_freq):
    utterance_field = data.Field(sequential=True,
                                 tokenize=lambda s: ast.literal_eval(s),
                                 # tokenize='spacy', # what's the difference?
                                 fix_length=15,
                                 batch_first=False,
                                 use_vocab=True,
                                 is_target=True,
                                 lower=True,
                                 unk_token='<unk>',
                                 pad_token='<pad>',
                                 init_token='<start>',
                                 eos_token='<eos>')

    train_val_fields = [
        ('utterance', utterance_field)
    ]

    train_dataset, val_dataset = data.TabularDataset.splits(path=data_path,
                                                            format='csv',
                                                            train=train_source,
                                                            validation=val_source,
                                                            fields=train_val_fields,
                                                            skip_header=True) # what about seed?

    # specify the path to the localy saved vectors
    # vec = vocab.Vectors(f'glove.6B.{WORD_EMBEDDING_SIZE}d.txt',
    #                     GLOVE_DIR)  # download here: https://nlp.stanford.edu/projects/glove/
    vec = GloVe('6B', dim=emb_size, cache=data_path)
    utterance_field.build_vocab(train_dataset, vectors=vec, min_freq=min_freq)
    # utterance_field.build_vocab(train_dataset, val_dataset, vectors=vec, min_freq=min_freq)

    # print(utterance_field.vocab.vectors[utterance_field.vocab.stoi['the']])

    train_loader, val_loader = data.BucketIterator.splits(datasets=(train_dataset, val_dataset),
                                                          batch_sizes=(train_batch_size, val_batch_size),
                                                          repeat=False, sort=False)

    datasets = (train_dataset, val_dataset)
    data_loaders = (train_loader, val_loader)
    return datasets, data_loaders, utterance_field


def get_secondary_loader(utterance_field, source, batch_size):
    dataset = data.TabularDataset(path=source,
                                format='csv',
                                fields=[('utterance', utterance_field)],
                                skip_header=True) # what about seed?

    loader = data.BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            train=False,
            repeat=False, 
            sort=False)
    iterator = BatchGenerator(loader, 'utterance')
    return iterator