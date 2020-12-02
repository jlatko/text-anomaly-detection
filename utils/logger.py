import os
import json
import numpy as np
from datetime import datetime
import torch
import shutil
import pickle


class Logger:
    def __init__(self, model_name, model, optimizer, train_eval, val_eval, data_path):
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.train_eval = train_eval
        self.val_eval = val_eval
        self.date = datetime.now()
        self.val_losses_desc = {}
        self.train_losses_desc = {}
        self.anomaly_roc = {}
        self.generated_sentences = {}
        self.anomaly_scores = {}

        dir_path = f"{data_path}\\models\\{self.model_name}\\runs\\{self.date.strftime('%m-%d')}\\{self.date.strftime('%H-%M')}"
        if os.path.isdir(dir_path):
            new_dir = str(int(max(os.listdir(dir_path))) + 1)
        else:
            new_dir = "1"
        os.makedirs(f"{dir_path}\\{new_dir}")
        self.dir_path = f"{dir_path}\\{new_dir}"

    def save_tags_and_script(self, tags):
        with open(f'{self.dir_path}\\tags.txt', 'w') as fp:
            fp.write(tags)
        try:
            shutil.copy('experiment.py', self.dir_path)
        except Exception as e:
            print('saving experiment script failed')
            print(e)

    def save_progress(self, epoch):
        recon_loss, kl_loss, loss = self.train_eval.mean_losses()
        self.train_losses_desc[epoch] = {
            "name": 'train',
            "epoch": epoch,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "loss": loss,
            # "KL": self.train_eval.kl_losses,
            # "Recon": self.train_eval.recon_losses,
            # "Loss": self.train_eval.losses,
            # "means_Loss_KL_Recon":
        }
        recon_loss, kl_loss, loss = self.val_eval.mean_losses()
        self.val_losses_desc[epoch] = {
            "name": 'val',
            "epoch": epoch,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "loss": loss,
            # "KL": self.val_eval.kl_losses,
            # "Recon": self.val_eval.recon_losses,
            # "Loss": self.val_eval.losses,
            # "means_Loss_KL_Recon": self.val_eval.mean_losses()
        }

        # with open(f'{self.dir_path}/train_losses.json', 'w') as fp:
        #     json.dump(train_losses, fp, indent=2)
        # with open(f'{self.dir_path}/val_losses.json', 'w') as fp:
        #     json.dump(val_losses, fp, indent=2)
        with open(f'{self.dir_path}\\train_losses_desc.json', 'w') as fp:
            json.dump(self.train_losses_desc, fp, indent=2)
        with open(f'{self.dir_path}\\val_losses_desc.json', 'w') as fp:
            json.dump(self.val_losses_desc, fp, indent=2)

        # SAVING MODEL
        torch.save(self.model.state_dict(), f"{self.dir_path}\\model.pth")
        # SAVING MODEL PARAMETERS
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, f"{self.dir_path}\\parameters.pth")

    def save_and_log_anomaly(self, epoch, auc, auc_kl, auc_recon, recon, kl):
        print(f'Anomaly detection ROC AUC - recon: {auc_recon:.3f} | KL: {auc_kl:.3f} |  recon+KL: {auc:.3f}')
        self.anomaly_roc[epoch] = {
            'auc': auc,
            'auc_kl': auc_kl,
            'auc_recon': auc_recon,
        }
        with open(f'{self.dir_path}\\anomaly.json', 'w') as fp:
            json.dump(self.anomaly_roc, fp, indent=2)

        self.anomaly_scores[epoch] = {
            'recon': recon,
            'kl': kl,
        }
        with open(f'{self.dir_path}\\anomaly_scores.pickle', 'wb') as fp:
            pickle.dump(self.anomaly_scores, fp)

    def save_and_log_sentences(self, epoch, rec_train, rec_val, rec_prior):
        print('train reconstruction (no dropout)')
        for d in rec_train:
            print('T: ', d['target'])
            print('R: ', d['reconstruction'])
            print('S: ', d['z_sample'])
        print('val reconstruction')
        for d in rec_val:
            print('T: ', d['target'])
            print('R: ', d['reconstruction'])
            print('S: ', d['z_sample'])
        print('Random sentences from prior')
        for txt in rec_prior:
            print(txt)
        self.generated_sentences[epoch] = {
            'train': rec_train,
            'val': rec_val,
            'prior': rec_prior
        }
        with open(f'{self.dir_path}\\generated_sentences.json', 'w') as fp:
            json.dump(self.generated_sentences, fp, indent=2)
