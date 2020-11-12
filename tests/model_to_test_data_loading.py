from torch import nn, optim
import matplotlib.pyplot as plt


from torchtext import data

from datasets.batch_generator import BatchGenerator
from datasets.data_loader import init_data_loading
from hyperparams import WORD_EMBEDDING_SIZE, LEARNING_RATE, BATCH_SIZE, EPOCHS

(train_dataset, val_dataset), (train_loader, val_loader), utterance_field = init_data_loading()

vocab_size = len(utterance_field.vocab)


class ExampleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_vec):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec)
        self.emb.weight.requires_grad = False

        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

    def embed(self, sequence):
        return self.emb(sequence)

    def forward(self, sequence):
        embs = self.embed(sequence)
        x, _ = self.gru(embs)
        x = nn.ReLU()(x)
        return x


if __name__ == "__main__":
    traindl, valdl = data.BucketIterator.splits(datasets=(train_dataset, val_dataset),
                                                batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                repeat=False)

    train_batch_it = BatchGenerator(traindl, 'utterance')
    val_batch_it = BatchGenerator(valdl, 'utterance')

    net = ExampleModel(vocab_size, WORD_EMBEDDING_SIZE, train_dataset.fields['utterance'].vocab.vectors)

    opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LEARNING_RATE)
    # filter makes sense, as we are not optimizing the Embedding layer!
    criterion = nn.MSELoss()

    net.train()
    max_epochs = 5
    e = 0
    losses = []
    for e in range(EPOCHS):
        for batch_input in iter(train_batch_it):
            batch_output_embeddings = net(batch_input)
            batch_input_embeddings = net.embed(batch_input)
            loss = criterion(batch_output_embeddings, batch_input_embeddings)
            losses.append(loss)
            batch_output_embeddings.sum().backward()
        plt.plot(list(range(len(losses))), losses)
    plt.show()