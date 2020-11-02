
class AnomalyEvaluator:
    def __init__(self):
        self.preds = [] # whatever, change it later

    def update(self, y, y_prob):
        # TODO: ???
        raise NotImplementedError

    def change_name(self):
        # TODO: some function that returns 
        # some statistics after the whole epoch (train or eval)
        raise NotImplementedError