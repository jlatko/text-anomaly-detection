import torch


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def print_random_sentences(model, utterance_field,  n=5):
    sentences = model.generate_sentences(batch_size=n).numpy()
    for sent in sentences:
        print(' '.join([utterance_field.vocab.itos[i] for i in sent]))