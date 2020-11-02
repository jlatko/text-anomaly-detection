from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Text dataset"""

    def __init__(self, source):
        """
        Args:
            source ():
        """
        self.source = source
        self.data = self.load_data(source)

    def load_data(self, source):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError