import sys

import numpy as np
import progressbar
import torch

from datasets.batch_generator import BatchGenerator
from datasets.data_loader import init_data_loading
from models.rnn_vae import RNN_VAE


def setup_model_and_dataloading(train_source, val_source, batch_size, data_path,
                                word_embedding_size, lr, min_freq, model_kwargs):

    ((train_dataset, val_dataset),
     (train_loader, val_loader),
     utterance_field)  = init_data_loading(data_path=data_path,
                                           train_batch_size=batch_size,
                                           val_batch_size=batch_size,
                                           emb_size=word_embedding_size,
                                           train_source=train_source,
                                           val_source=val_source,
                                           min_freq=min_freq)

    vocab_size = len(utterance_field.vocab)

    train_batch_it = BatchGenerator(train_loader, 'utterance')
    val_batch_it = BatchGenerator(val_loader, 'utterance')


    model = RNN_VAE(
            vocab_size,
            unk_idx=utterance_field.vocab.stoi['<unk>'],
            pad_idx=utterance_field.vocab.stoi['<pad>'],
            start_idx=utterance_field.vocab.stoi['<start>'],
            eos_idx=utterance_field.vocab.stoi['<eos>'],
            pretrained_embeddings=utterance_field.vocab.vectors,
            gpu=torch.cuda.is_available(),
            **model_kwargs
        )
    opt = torch.optim.Adam(model.vae_params, lr=lr)
    return train_batch_it, val_batch_it, model, opt, utterance_field


def get_kl_weight(epoch):
    return float(1/(1+np.exp(-0.2*(epoch-30))))

def get_train_pbar(epoch):
    widgets = [progressbar.FormatLabel(f'Epoch {epoch:3d} | Batch '),
               progressbar.SimpleProgress(), ' | ',
               progressbar.Percentage(), ' | ',
               progressbar.FormatLabel(f'Loss N/A'), ' | ',
               progressbar.Timer(), ' | ',
               progressbar.ETA()]

    return progressbar.ProgressBar(widgets=widgets, fd=sys.stdout)


def train_step(epoch, model, train_eval, train_batch_it, opt):
    kld_weight = get_kl_weight(epoch)
    print(f'KL weight: {kld_weight}')
    # TRAINING
    model.train()
    train_eval.reset()

    pbar = get_train_pbar(epoch=epoch)
    for batch_input in pbar(train_batch_it):
        # run model
        res = model.forward(batch_input)

        # compute loss and backpropagate 
        recon_loss = res['recon_loss']
        kl_loss = res['kl_loss']
        loss = recon_loss + kld_weight * kl_loss
        loss.backward()

         # if frozen: leave gradient only for special tokens 
        model.mask_embedding_grad()

        # optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params, 5) 
        opt.step()
        opt.zero_grad()

        # evaluation and logging
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