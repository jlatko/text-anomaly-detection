import os
import json
import numpy as np
from datetime import datetime
import torch

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
        self.generated_sentences = {}

        dir_path = f"{data_path}/models/{self.model_name}/runs/{self.date.strftime('%m-%d')}/{self.date.strftime('%H:%M')}"
        if os.path.isdir(dir_path):
            new_dir = str(int(max(os.listdir(dir_path))) + 1)
        else:
            new_dir = "1"
        os.makedirs(f"{dir_path}/{new_dir}")
        self.dir_path = f"{dir_path}/{new_dir}"

    def save_progress(self, epoch):
        # to chyba niepotrzebne
        # train_losses = {
        #     "Loss": self.train_eval.kl_losses,
        #     "KL": self.train_eval.recon_losses,
        #     "Recon": self.train_eval.losses
        # }
        # val_losses = {
        #     "Loss": self.val_eval.kl_losses,
        #     "KL": self.val_eval.recon_losses,
        #     "Recon": self.val_eval.losses
        # }
        # chyba wystarczy mean loss epoki, nie potrzebujemy per batch
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
        with open(f'{self.dir_path}/train_losses_desc.json', 'w') as fp:
            json.dump(self.train_losses_desc, fp, indent=2)
        with open(f'{self.dir_path}/val_losses_desc.json', 'w') as fp:
            json.dump(self.val_losses_desc, fp, indent=2)

        # SAVING MODEL
        torch.save(self.model.state_dict(), f"{self.dir_path}/model.pth")
        # SAVING MODEL PARAMETERS
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, f"{self.dir_path}/parameters.pth")

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
        with open(f'{self.dir_path}/generated_sentences.json', 'w') as fp:
            json.dump(self.generated_sentences, fp, indent=2)

