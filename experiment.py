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
from utils.model_utils import print_random_sentences

torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
np.random.seed(12)

# solve setup nicely
N_EPOCHS = 100
BATCH_SIZE = 16
# NUM_WORKERS = 2 # threads for dataloading
LEARNING_RATE = 1e-3
WORD_EMBEDDING_SIZE = 50
Z_SIZE = 64
RNN_HIDDEN = 64

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
            vocab_size, RNN_HIDDEN, Z_SIZE,
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
    opt = torch.optim.Adam(model.vae_params, lr=LEARNING_RATE)
    return train_batch_it, val_batch_it, model, opt, utterance_field

def get_kl_weight(epoch):
    return float(1/(1+np.exp(-0.2*(epoch-30))))


def train():
    train_batch_it, val_batch_it, model, opt, utterance_field = setup_model_and_dataloading()

    print(model)
    train_eval = VAEEvaluator()
    val_eval = VAEEvaluator()

    for epoch in range(N_EPOCHS):
        kld_weight = get_kl_weight(epoch)
        print(f'KL weight: {kld_weight}')
        # TRAINING
        model.train()
        train_eval.reset()
        widgets = [progressbar.FormatLabel(f'Epoch {epoch:3d} | Batch '),
                   progressbar.SimpleProgress(), ' | ',
                   progressbar.Percentage(), ' | ',
                   progressbar.FormatLabel(f'Loss N/A'), ' | ',
                   progressbar.Timer(), ' | ',
                   progressbar.ETA()]

        pbar = progressbar.ProgressBar(widgets=widgets, fd=sys.stdout)
        # n = 0
        for batch_input in pbar(train_batch_it):
            # n += 1
            recon_loss, kl_loss = model.forward(batch_input)
            loss = recon_loss + kld_weight * kl_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params, 5)
            opt.step()
            opt.zero_grad()
            train_eval.update(recon_loss.item(), kl_loss.item(), loss.item())
            e_recon_loss, e_kl_loss, e_loss = train_eval.mean_losses()
            pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) {e_loss:.2f} / {loss.item():.2f} || '
                                                      f'KL {e_kl_loss:.2f} / {kl_loss.item():.2f} || '
                                                      f'Recon {e_recon_loss:.2f} / {recon_loss.item():.2f}')
            # if n == 3000:
            #     break

        e_recon_loss, e_kl_loss, e_loss = train_eval.mean_losses()
        print(f'EPOCH {epoch} TRAIN Loss: {e_loss:.2f} || '
              f'KL {e_kl_loss:.2f} || '
              f'Recon {e_recon_loss:.2f} ')

        # EVALUATION
        model.eval()

        print_random_sentences(model, utterance_field)
        val_eval.reset()
        pbar = progressbar.ProgressBar(fd=sys.stdout)
        for batch_input in pbar(val_batch_it):
            recon_loss, kl_loss = model.forward(batch_input)
            val_eval.update(recon_loss.item(), kl_loss.item(), 0)
        val_recon_loss, val_kl_loss, _ = val_eval.mean_losses()
        print(f'EPOCH {epoch} VAL '
              f'KL {val_kl_loss:.2f} || '
              f'Recon {val_recon_loss:.2f} ')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Text VAE'
    )
    args = parser.parse_args()
    train()



