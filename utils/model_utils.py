import torch

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

def to_cpu(x):
    return x.cpu() if torch.cuda.is_available() else x

def print_random_sentences(model, utterance_field,  n=5):
    sentences = to_cpu(model.generate_sentences(batch_size=n)).numpy()
    for sent in sentences:
        print(' '.join([utterance_field.vocab.itos[i] for i in sent]))

def print_reconstructed_sentences(target, reconstruction, utterance_field, n=5):
    target = target.numpy()
    reconstruction = reconstruction.numpy().argmax(axis=-1)
    for t, r in zip(target[:n], reconstruction[:n]):
        print(' '.join([utterance_field.vocab.itos[i] for i in t]))
        print(' '.join([utterance_field.vocab.itos[i] for i in r]))