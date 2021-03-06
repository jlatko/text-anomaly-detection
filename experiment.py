import json
import os

import numpy as np
import torch
from sacred import Experiment

from datasets.data_loader import get_secondary_loader
from utils.logger import Logger

from evaluators.vae_evaluator import VAEEvaluator
from utils.experiment_utils_lm import setup_model_and_dataloading_lm, train_step_lm, val_step_lm, detect_anomalies_lm
from utils.experiment_utils import setup_model_and_dataloading, train_step, val_step, detect_anomalies
from utils.model_utils import get_random_sentences, get_reconstructed_sentences, to_cpu, get_sentences_lm
from utils.corpus import get_corpus

torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
np.random.seed(12)

ex = Experiment('text_vae')


@ex.config
def default_config():
    # data_path = 'data/'
    data_path = '../text-anomaly-detection/data'
    tags = 'LSTM parliament vs friends scale=0.2 bs=32 \n'
    #source = 'friends-corpus'
    #source = 'supreme-corpus'
    ood_source = 'friends-corpus'
    source = 'parliament-corpus'
    # source = 'IMDB Dataset.csv'
    split_sentences = True
    punct = False
    to_ascii = True
    min_len = 3
    max_len = 15
    test_size = 0.1
    text_field = 'text'
    batch_size = 32
    word_embedding_size = 50
    optimizer_kwargs = {
        'lr': 1e-3
    }
    n_epochs = 100
    print_every = 1
    subsample_rows = None  # for testing
    subsample_rows_ood = None
    min_freq = 1
    decode=False
    model_kwargs = {
        'set_other_to_random': False,
        'set_unk_to_random': True,
        'decode_with_embeddings': decode, # [False, 'cosine', 'cdist']
        'h_dim': 256,
        'z_dim': 256,
        # 'p_word_dropout': 0.5,
        'p_word_dropout': 0.3,
        'max_sent_len':  max_len,
        'freeze_embeddings': False,
        'rnn_dropout': 0.3,
        'mask_pad': True,
    }
    kl_kwargs = {
        'cycles': 4,
        'scale': 0.2
    }


@ex.capture
def train(source, batch_size, word_embedding_size, model_kwargs, optimizer_kwargs, kl_kwargs,
          n_epochs, print_every, split_sentences, punct, to_ascii, min_freq,
          min_len, max_len, test_size, text_field, subsample_rows, data_path,
          ood_source, subsample_rows_ood, tags):
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
    _, _, _, ood_source_csv = get_corpus(source=ood_source,
                                         split_sentences=split_sentences,
                                         punct=punct,
                                         to_ascii=to_ascii,
                                         data_path=data_path,
                                         min_len=min_len,
                                         max_len=max_len,
                                         # test_size=1,
                                         test_size=test_size,  # so it's always the same for ood and val
                                         text_field=text_field,
                                         subsample_rows=subsample_rows_ood)

    ood_it = get_secondary_loader(utterance_field, os.path.join(data_path, ood_source_csv), batch_size=batch_size)
    # TODO: consider including secondary dataset in "build vocab"

    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    logger = Logger(model_name="RNN", model=model, optimizer=opt,
                    train_eval=train_eval, val_eval=val_eval, data_path=data_path)
    # TODO: make it into something nicer
    tags += '\n'.join([
        source,
        ood_source,
        json.dumps(kl_kwargs),
        json.dumps(model_kwargs)
    ])
    logger.save_tags_and_script(tags)

    for epoch in range(n_epochs):
        # Train
        train_step(epoch, model, train_eval, train_batch_it, opt, n_epochs, kl_kwargs)
        train_eval.log_and_save_progress(epoch, 'train')

        # Val
        val_step(model, val_eval, val_batch_it, utterance_field)
        val_eval.log_and_save_progress(epoch, 'val')

        # save progress to file
        logger.save_progress(epoch)

        if (epoch + 1) % print_every == 0:
            # generate sentences
            model.eval()
            example_batch = next(iter(train_batch_it))
            rec_train = get_reconstructed_sentences(model, example_batch, utterance_field)

            model.eval()
            example_batch = next(iter(val_batch_it))
            rec_val = get_reconstructed_sentences(model, example_batch, utterance_field)

            rec_prior = get_random_sentences(model, utterance_field)

            # simple anomaly detection ROC AUC scores
            auc, auc_kl, auc_recon, recon, kl = detect_anomalies(model, val_batch_it, ood_it, kl_weight=0.1)

            logger.save_and_log_anomaly(epoch, auc, auc_kl, auc_recon, recon, kl)

            # save and log generated
            logger.save_and_log_sentences(epoch, rec_train, rec_val, rec_prior)


@ex.capture
def train_lm(source, batch_size, word_embedding_size, model_kwargs, optimizer_kwargs, kl_kwargs,
             n_epochs, print_every, split_sentences, punct, to_ascii, min_freq,
             min_len, max_len, test_size, text_field, subsample_rows, data_path,
             ood_source, subsample_rows_ood, tags):
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
    ) = setup_model_and_dataloading_lm(train_source=train_source,
                                       val_source=val_source,
                                       batch_size=batch_size,
                                       data_path=data_path,
                                       word_embedding_size=word_embedding_size,
                                       optimizer_kwargs=optimizer_kwargs,
                                       min_freq=min_freq,
                                       model_kwargs=model_kwargs)

    # prepare anomaly data
    _, _, _, ood_source_csv = get_corpus(source=ood_source,
                                         split_sentences=split_sentences,
                                         punct=punct,
                                         to_ascii=to_ascii,
                                         data_path=data_path,
                                         min_len=min_len,
                                         max_len=max_len,
                                         # test_size=1,
                                         test_size=test_size,  # so it's always the same for ood and val
                                         text_field=text_field,
                                         subsample_rows=subsample_rows_ood)

    ood_it = get_secondary_loader(utterance_field, os.path.join(data_path, ood_source_csv), batch_size=batch_size)
    # TODO: consider including secondary dataset in "build vocab"

    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    logger = Logger(model_name="RNN", model=model, optimizer=opt,
                    train_eval=train_eval, val_eval=val_eval, data_path=data_path)
    # TODO: make it into something nicer
    tags += '\n'.join([
        source,
        ood_source,
        json.dumps(kl_kwargs),
        json.dumps(model_kwargs)
    ])
    logger.save_tags_and_script(tags)

    for epoch in range(n_epochs):
        # Train
        train_step_lm(epoch, model, train_eval, train_batch_it, opt, n_epochs, kl_kwargs)
        train_eval.log_and_save_progress(epoch, 'train')

        # Val
        val_step_lm(model, val_eval, val_batch_it)
        val_eval.log_and_save_progress(epoch, 'val')

        # save progress to file
        logger.save_progress(epoch)

        if (epoch + 1) % print_every == 0:
            # generate sentences
            model.eval()
            example_batch = next(iter(train_batch_it))
            rec_train = get_sentences_lm(model, example_batch, utterance_field)

            model.eval()
            example_batch = next(iter(val_batch_it))
            rec_val = get_sentences_lm(model, example_batch, utterance_field)

            # simple anomaly detection ROC AUC scores
            auc, auc_recon, recon = detect_anomalies_lm(model, val_batch_it, ood_it)
            logger.save_and_log_anomaly(epoch, auc, 0, auc_recon, recon, 0)

            # save and log generated
            logger.save_and_log_sentences(epoch, rec_train, rec_val, [])


@ex.automain
def main():
    train()
