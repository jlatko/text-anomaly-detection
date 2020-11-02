
class VAEEvaluator:
    def __init__(self):
        self.score = 0

    def update(self, x, x_pred, loss):
        # TODO: what about z?
        raise NotImplementedError

    def change_name(self):
        # TODO: some function that returns 
        # some statistics after the whole epoch (train or eval)
        raise NotImplementedError
