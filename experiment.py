import sys

import numpy as np
import progressbar
import torch
from sacred import Experiment

from evaluators.vae_evaluator import VAEEvaluator
from utils.experiment_utils import setup_model_and_dataloading, get_kl_weight, get_train_pbar
from utils.model_utils import print_random_sentences, print_reconstructed_sentences, to_cpu

torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
np.random.seed(12)

ex = Experiment('text_vae')


@ex.config
def default_config():
    train_source = 'tiny_train.csv' # for debugging
    # train_source = 'traindf.csv'
    val_source = 'tiny_val.csv' # for debugging
    # val_source = 'valdf.csv'
    batch_size = 16
    word_embedding_size = 50
    rnn_hidden = 128
    z_size = 128
    lr = 1e-3
    n_epochs = 100
    print_every = 10

def train_step(epoch, model, train_eval, train_batch_it, opt):
    kld_weight = get_kl_weight(epoch)
    print(f'KL weight: {kld_weight}')
    # TRAINING
    model.train()
    train_eval.reset()

    pbar = get_train_pbar(epoch=epoch)
    for batch_input in pbar(train_batch_it):
        res = model.forward(batch_input)
        recon_loss = res['recon_loss']
        kl_loss = res['kl_loss']
        loss = recon_loss + kld_weight * kl_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params, 5)  # tODO use this?
        opt.step()
        opt.zero_grad()
        train_eval.update(recon_loss.item(), kl_loss.item(), loss.item())
        e_recon_loss, e_kl_loss, e_loss = train_eval.mean_losses()
        pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) {e_loss:.2f} / {loss.item():.2f} || '
                                                  f'KL {e_kl_loss:.2f} / {kl_loss.item():.2f} || '
                                                  f'Recon {e_recon_loss:.2f} / {recon_loss.item():.2f}')

def val_step(model, val_eval, val_batch_it, utterance_field):
    # EVALUATION
    model.eval()
    val_eval.reset()
    pbar = progressbar.ProgressBar(fd=sys.stdout)
    for batch_input in pbar(val_batch_it):
        res = model.forward(batch_input)
        recon_loss = res['recon_loss']
        kl_loss = res['kl_loss']
        val_eval.update(recon_loss.item(), kl_loss.item(), np.nan)

@ex.capture
def train(train_source, val_source, batch_size, word_embedding_size, rnn_hidden, z_size, lr, n_epochs, print_every):
    (
        train_batch_it, val_batch_it, model, opt, utterance_field
    ) = setup_model_and_dataloading(train_source=train_source,
                                    val_source=val_source,
                                    batch_size=batch_size,
                                    word_embedding_size=word_embedding_size,
                                    rnn_hidden=rnn_hidden,
                                    z_size=z_size,
                                    lr=lr)

    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    for epoch in range(n_epochs):
        # Train
        train_step(epoch, model, train_eval, train_batch_it, opt)
        train_eval.log_and_save_progress(epoch, 'train') # TODO# print sentences


        # Val
        val_step(model, val_eval, val_batch_it, utterance_field)
        val_eval.log_and_save_progress(epoch, 'val')

        if (epoch+1) % print_every == 0:
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

@ex.automain
def main():
    train()