import sys

import numpy as np
import progressbar
import torch

from datasets.batch_generator import BatchGenerator
from datasets.data_loader import init_data_loading
from models.rnn_vae import RNN_VAE


def setup_model_and_dataloading(train_source, val_source, batch_size,
                                word_embedding_size, rnn_hidden, z_size, lr):

    ((train_dataset, val_dataset),
     (train_loader, val_loader),
     utterance_field)  = init_data_loading(data_path='data/',
                                           train_batch_size=batch_size,
                                           val_batch_size=batch_size,
                                           emb_size=word_embedding_size,
                                           train_source=train_source,
                                           val_source=val_source)

    vocab_size = len(utterance_field.vocab)

    train_batch_it = BatchGenerator(train_loader, 'utterance')
    val_batch_it = BatchGenerator(val_loader, 'utterance')


    model = RNN_VAE(
            vocab_size, rnn_hidden, z_size,
            unk_idx=utterance_field.vocab.stoi['<unk>'],
            pad_idx=utterance_field.vocab.stoi['<pad>'],
            start_idx=utterance_field.vocab.stoi['<start>'],
            eos_idx=utterance_field.vocab.stoi['<eos>'],
            max_sent_len=15,
            p_word_dropout=0.3,
            pretrained_embeddings=utterance_field.vocab.vectors,
            freeze_embeddings=True,
            gpu=torch.cuda.is_available()
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

