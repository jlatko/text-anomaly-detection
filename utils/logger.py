import os
import json
import numpy as np
from datetime import datetime


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

        dir_path = f"{data_path}/models/{self.model_name}/runs/{self.date.strftime('%m-%d')}/{self.date.strftime('%H:%M')}"
        if os.path.isdir(dir_path):
            new_dir = str(int(max(os.listdir(dir_path))) + 1)
        else:
            new_dir = "1"
        os.mkdir(f"{dir_path}/{new_dir}")
        dir_path = f"{dir_path}/{new_dir}"

    def save_progress(self, epoch):
        train_losses = {
            "Loss": self.train_eval.kl_losses,
            "KL": self.train_eval.recon_losses,
            "Recon": self.train_eval.losses
        }
        val_losses = {
            "Loss": self.val_eval.kl_losses,
            "KL": self.val_eval.recon_losses,
            "Recon": self.val_eval.losses
        }
        self.train_losses_desc[f'{epoch}'] = {
            "name": 'train',
            "epoch": epoch,
            "Loss": [loss for loss in self.train_eval.kl_losses],
            "KL": [loss for loss in self.train_eval.recon_losses],
            "Recon": [loss for loss in self.train_eval.losses],
            "means_Loss_KL_Recon": [val for val in self.train_eval.mean_losses()]
        }
        self.val_losses_desc[f'{epoch}'] = {
            "name": 'val',
            "epoch": epoch,
            "Loss": [loss for loss in selfval_eval.kl_losses],
            "KL": [loss for loss in self.val_eval.recon_losses],
            "Recon": [loss for loss in self.val_eval.losses],
            "means_Loss_KL_Recon": [val for val in self.val_eval.mean_losses()]
        }

        with open(f'{dir_path}/train_losses.json', 'w') as fp:
            json.dump(train_losses, fp, indent=2)
        with open(f'{dir_path}/val_losses.json', 'w') as fp:
            json.dump(val_losses, fp, indent=2)
        with open(f'{dir_path}/train_losses_desc.json', 'w') as fp:
            json.dump(self.train_losses_desc, fp, indent=2)
        with open(f'{dir_path}/val_losses_desc.json', 'w') as fp:
            json.dump(self.val_losses_desc, fp, indent=2)

        # SAVING MODEL
        torch.save(self.model, f"{dir_path}/model.pth")
        # SAVING MODEL PARAMETERS
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, f"{dir_path}/parameters.pth")