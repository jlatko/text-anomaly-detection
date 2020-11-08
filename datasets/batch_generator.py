class BatchGenerator:
    def __init__(self, dl, x_field):
        self.dl, self.x_field = dl, x_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for _batch in self.dl:
            X = getattr(_batch, self.x_field)
            yield X
