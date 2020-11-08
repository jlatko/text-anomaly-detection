
class VAEEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.kl_loss = 0
        self.recon_loss = 0


    def update(self, recon_loss, kl_loss, loss):
        pass

    def change_name(self):
        # TODO: some function that returns 
        # some statistics after the whole epoch (train or eval)
        raise NotImplementedError
