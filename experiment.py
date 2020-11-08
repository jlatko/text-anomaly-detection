import argparse
import sys

import numpy as np
import progressbar
import torch
from torch.utils.data import DataLoader
from torchtext import data
from tqdm import tqdm

from datasets.batch_generator import BatchGenerator
from datasets.data_loader import init_data_loading
from datasets.sst_dataset import SST_Dataset
from evaluators.vae_evaluator import VAEEvaluator
from models.rnn_vae import RNN_VAE
from models.vae import VAE

torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
np.random.seed(12)

# TODO: would be nice to have some framework or lib for the experiment, 
# but for now let's have anything to test on
#

# solve setup nicely
N_EPOCHS = 10  
BATCH_SIZE = 16
NUM_WORKERS = 2 # threads for dataloading
LEARNING_RATE = 1e-3
WORD_EMBEDDING_SIZE = 50
EPOCHS = 5
mb_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2

def setup_model_and_dataloading():
    # TODO: load properly

    ((train_dataset, val_dataset),
     (train_loader, val_loader),
     utterance_field)  = init_data_loading(data_path='data/',
                                           train_batch_size=BATCH_SIZE,
                                           val_batch_size=BATCH_SIZE,
                                           emb_size=WORD_EMBEDDING_SIZE)

    vocab_size = len(utterance_field.vocab)

    train_batch_it = BatchGenerator(train_loader, 'utterance')
    val_batch_it = BatchGenerator(val_loader, 'utterance')

    model = RNN_VAE(
            vocab_size, h_dim, z_dim, c_dim, p_word_dropout=0.3,
            pretrained_embeddings=utterance_field.vocab.vectors, freeze_embeddings=True,
            gpu=torch.cuda.is_available()
        )
    opt = torch.optim.Adam(model.vae_params, lr=LEARNING_RATE)

    return train_batch_it, val_batch_it, model, opt




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
    )

    parser.add_argument('--gpu', default=False, action='store_true',
                        help='whether to run in the GPU')
    parser.add_argument('--save', default=False, action='store_true',
                        help='whether to save model or not')

    args = parser.parse_args()

    train_batch_it, val_batch_it, model, opt = setup_model_and_dataloading()
    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()


    for epoch in range(N_EPOCHS):
        kld_weight = 1 # TODO: annealing
        # progressbar
        model.train()
        train_eval.reset()
        widgets = [progressbar.FormatLabel(f'Epoch {epoch:3d} | Batch '),
                   progressbar.SimpleProgress(), ' | ',
                   progressbar.Percentage(), ' | ',
                   progressbar.FormatLabel(f'Loss N/A'), ' | ',
                   progressbar.Timer(), ' | ',
                   progressbar.ETA()]

        pbar = progressbar.ProgressBar(widgets=widgets, fd=sys.stdout)
        for batch_input in pbar(train_batch_it):
            recon_loss, kl_loss = model.forward(batch_input)
            loss = recon_loss + kld_weight * kl_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params, 5)
            opt.step()
            opt.zero_grad()
            train_eval.update(recon_loss, kl_loss, loss)
            pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) / {loss.item():.2f} || '
                                                      f'KL {kl_loss.item():.2f} || '
                                                      f'Recon {recon_loss.item():.2f}')

            # TODO: log

        model.eval()
        val_eval.reset()
        for batch_input in iter(val_batch_it):
            # TODO: predict
            # TODO: evaluate
            # TODO: log
            pass

        # TODO: possibly anomaly detection (every n epochs?)

