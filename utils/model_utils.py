import torch

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

def to_cpu(x):
    return x.cpu() if torch.cuda.is_available() else x

def get_random_sentences(model, utterance_field,  n=5):
    sentences = to_cpu(model.generate_sentences(batch_size=n)).numpy()
    txt = []
    for sent in sentences:
        txt.append(' '.join([utterance_field.vocab.itos[i] for i in sent]))
    return txt

def get_reconstructed_sentences(model, batch, utterance_field, n=5):
    target = batch.numpy().transpose()
    res = model.forward(batch)
    reconstruction = to_cpu(res['y']).detach()
    reconstruction = reconstruction.numpy().argmax(axis=-1).transpose()
    txt = []
    for t, r, z in zip(target[:n], reconstruction[:n], res['z'][:n]):
        sample = model.sample_sentence(z, raw=True)[0]
        txt.append({
                'target': ' '.join([utterance_field.vocab.itos[i] for i in t]),
                'reconstruction': ' '.join([utterance_field.vocab.itos[i] for i in r]),
                'z_sample': ' '.join([utterance_field.vocab.itos[i] for i in sample])
            })
    return txt


