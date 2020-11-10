import numpy as np

class VAEEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.kl_losses = []
        self.recon_losses = []
        self.losses = []


    def update(self, recon_loss, kl_loss, loss):
        self.kl_losses.append(kl_loss)
        self.recon_losses.append(recon_loss)
        self.losses.append(loss)

    def mean_losses(self):
        return np.mean(self.recon_losses), np.mean(self.kl_losses), np.mean(self.losses)

    def log_and_save_progress(self, epoch, name):
        e_recon_loss, e_kl_loss, e_loss = self.mean_losses()
        print(f'EPOCH {epoch} {name.upper()} Loss: {e_loss:.2f} || '
              f'KL {e_kl_loss:.2f} || '
              f'Recon {e_recon_loss:.2f} ')