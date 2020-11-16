import sys

import numpy as np
import progressbar
import torch
from sacred import Experiment

from evaluators.vae_evaluator import VAEEvaluator
from utils.experiment_utils import setup_model_and_dataloading, train_step, val_step
from utils.model_utils import print_random_sentences, print_reconstructed_sentences, to_cpu
from utils.corpus import get_corpus

torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
np.random.seed(12)

ex = Experiment('text_vae')


@ex.config
def default_config():
    data_path = 'data/'
    source = 'friends-corpus'
    # source = 'IMDB Dataset.csv' 
    split_sentences = True
    punct = False
    to_ascii = True
    min_len = 3
    max_len = 15
    test_size = 0.1
    text_field = 'text'
    batch_size = 16
    word_embedding_size = 50
    lr = 1e-3
    n_epochs = 100
    print_every = 10
    subsample_rows = None  # for testing
    min_freq = 1
    model_kwargs = {
        'set_other_to_random': True,
        'set_unk_to_random': True,
        'decode_with_embeddings': True,
        'h_dim': 128,
        'z_dim': 128,
        'p_word_dropout': 0.3,
        'max_sent_len': max_len,
        'freeze_embeddings': False,
    }


@ex.capture
def train(source, batch_size, word_embedding_size, model_kwargs, lr,
          n_epochs, print_every, split_sentences, punct, to_ascii, min_freq,
          min_len, max_len, test_size, text_field, subsample_rows, data_path):
    # prepare/load data
    _, _, train_source, val_source = get_corpus(source=source,
                                                split_sentences=split_sentences,
                                                punct=punct,
                                                to_ascii=to_ascii,
                                                data_path=data_path,
                                                min_len=min_len,
                                                max_len=max_len,
                                                test_size=test_size,
                                                text_field=text_field,
                                                subsample_rows=subsample_rows)

    (
        train_batch_it, val_batch_it, model, opt, utterance_field
    ) = setup_model_and_dataloading(train_source=train_source,
                                    val_source=val_source,
                                    batch_size=batch_size,
                                    data_path=data_path,
                                    word_embedding_size=word_embedding_size,
                                    lr=lr, min_freq=min_freq,
                                    model_kwargs=model_kwargs)

    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    for epoch in range(n_epochs):
        # Train
        train_step(epoch, model, train_eval, train_batch_it, opt, n_epochs)
        train_eval.log_and_save_progress(epoch, 'train')  # TODO# print sentences

        # Val
        val_step(model, val_eval, val_batch_it, utterance_field)
        val_eval.log_and_save_progress(epoch, 'val')

        if (epoch + 1) % print_every == 0:
            # print sentences
            model.eval()
            print('train reconstruction (no dropout)')
            example_batch = next(iter(train_batch_it))
            res = model.forward(example_batch)
            print_reconstructed_sentences(example_batch, to_cpu(res['y']).detach(), utterance_field)

            model.eval()
            print('val reconstruction')
            example_batch = next(iter(val_batch_it))
            res = model.forward(example_batch)
            print_reconstructed_sentences(example_batch, to_cpu(res['y']).detach(), utterance_field)

            print('Random sentences from prior')
            print_random_sentences(model, utterance_field)
    train_eval.plot_training("Train")
    val_eval.plot_training("Val")


@ex.automain
def main():
    train()
