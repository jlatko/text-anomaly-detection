import os

import numpy as np
import torch
from sacred import Experiment

from datasets.data_loader import get_secondary_loader
from utils.logger import Logger

from evaluators.vae_evaluator import VAEEvaluator
from utils.experiment_utils import setup_model_and_dataloading, train_step, val_step, detect_anomalies
from utils.model_utils import get_random_sentences, get_reconstructed_sentences, to_cpu
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
    ood_source = 'supreme-corpus'
    # source = 'parliament-corpus'
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
    optimizer_kwargs = {
        'lr': 1e-3
    }
    n_epochs = 100
    print_every = 1
    subsample_rows = None  # for testing
    subsample_rows_ood = 10000
    min_freq = 1
    model_kwargs = {
        'set_other_to_random': False,
        'set_unk_to_random': True,
        'decode_with_embeddings': False, # [False, 'cosine', 'cdist']
        'h_dim': 256,
        'z_dim': 256,
        'p_word_dropout': 0.3,
        'max_sent_len':  max_len,
        'freeze_embeddings': True,
        'rnn_dropout': 0.3,
        'mask_pad': True,
    }
    kl_kwargs = {
        'cycles': 5,
        'scale': 0.01
    }

@ex.capture
def train(source, batch_size, word_embedding_size, model_kwargs, optimizer_kwargs, kl_kwargs,
          n_epochs, print_every, split_sentences, punct, to_ascii, min_freq,
          min_len, max_len, test_size, text_field, subsample_rows, data_path, 
          ood_source, subsample_rows_ood):
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
                                    optimizer_kwargs=optimizer_kwargs,
                                    min_freq=min_freq,
                                    model_kwargs=model_kwargs)

    # prepare anomaly data
    _, ood_source_csv = get_corpus(source=ood_source,
                                split_sentences=split_sentences,
                                punct=punct,
                                to_ascii=to_ascii,
                                data_path=data_path,
                                min_len=min_len,
                                max_len=max_len,
                                test_size=1,
                                text_field=text_field,
                                subsample_rows=subsample_rows_ood)

    # val_single_it = get_secondary_loader(utterance_field, os.path.join(data_path, val_source))
    # ood_it = get_secondary_loader(utterance_field, os.path.join(data_path, ood_source_csv))


    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    logger = Logger(model_name = "RNN", model = model, optimizer = opt,
            train_eval = train_eval, val_eval = val_eval, data_path=data_path)

    for epoch in range(n_epochs):
        # Train
        train_step(epoch, model, train_eval, train_batch_it, opt, n_epochs, kl_kwargs)
        train_eval.log_and_save_progress(epoch, 'train')

        # Val
        val_step(model, val_eval, val_batch_it, utterance_field)
        val_eval.log_and_save_progress(epoch, 'val')

        # save progress to file
        logger.save_progress(epoch)

        if (epoch+1) % print_every == 0:
            # generate sentences
            model.eval()
            example_batch = next(iter(train_batch_it))
            rec_train = get_reconstructed_sentences(model, example_batch, utterance_field)

            model.eval()
            example_batch = next(iter(val_batch_it))
            rec_val = get_reconstructed_sentences(model, example_batch, utterance_field)

            rec_prior = get_random_sentences(model, utterance_field)

            # save and log generated
            logger.save_and_log_sentences(epoch, rec_train, rec_val, rec_prior)

            # detect_anomalies(model, val_single_it, ood_it)
@ex.automain
def main():
    train()