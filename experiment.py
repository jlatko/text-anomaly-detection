import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.text_dataset import TextDataset
from models.vae import VAE

torch.manual_seed(12)
# torch.cuda.manual_seed(12)
np.random.seed(12)

# TODO: would be nice to have some framework or lib for the experiment, 
# but for now let's have anything to test on
#

# solve setup nicely
N_EPOCHS = 10  
BATCH_SIZE = 16
NUM_WORKERS = 2 # threads for dataloading

def setup_model():
    # TODO: load properly
    latent_features = 128
    input_shape = (256, )
    return VAE(input_shape=input_shape, latent_features=latent_features)

def setup_dataloaders():
    train_source = 'something'
    val_source = 'something else'
    # test_source = 'do we need it?'

    # maybe a common factory could return those two, will see
    train_dataset = TextDataset(source=train_source)
    val_dataset = TextDataset(source=val_source)
    # test_dataset = TextDataset(source=test_source)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


if __name__ == '__main__':
    model = setup_model()

    train_loader, val_loader = setup_dataloaders()

    for epoch in range(N_EPOCHS):
        # progressbar
        for X in train_loader:
            # TODO: train
            # TODO: evaluate
            # TODO: log
            pass

        for X in val_loader: 
            # TODO: predict
            # TODO: evaluate
            # TODO: log
            pass

        # TODO: possibly anomaly detection (every n epochs?)

