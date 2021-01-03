import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.model_utils import to_gpu


def cosine_sim_matrix(a, b, eps=1e-8):
    """
    cosine similarity
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class RNN_LanguageModel(nn.Module):
    def __init__(self, n_vocab, h_dim=128,
                 unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15,
                 use_input_embeddings=True, set_other_to_random=False,
                 set_unk_to_random=True, decode_with_embeddings=True,
                 rnn_dropout=0.3, mask_pad=True,
                 pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_LanguageModel, self).__init__()

        self.PAD_IDX = pad_idx
        self.UNK_IDX = unk_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len
        self.freeze_embeddings = freeze_embeddings

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.use_input_embeddings = use_input_embeddings  # in case we go for sentence embeddings
        self.decode_with_embeddings = decode_with_embeddings
        self.gpu = gpu

        if mask_pad:
            self.weights = np.ones(n_vocab)
            self.weights[self.PAD_IDX] = 0
            self.weights = to_gpu(torch.tensor(self.weights.astype(np.float32)))
        else:
            self.weights = None

        """
        Word embeddings layer
        """
        if self.use_input_embeddings:
            if pretrained_embeddings is None:
                self.emb_dim = h_dim
                self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
            else:
                self.emb_dim = pretrained_embeddings.size(1)
                self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

                # Set pretrained embeddings
                self.word_emb.weight.data.copy_(pretrained_embeddings)
        else:
            raise NotImplementedError('not supported yet')

        if set_unk_to_random:
            self.word_emb.weight[self.UNK_IDX].data.copy_(torch.tensor(np.random.randn(self.emb_dim)))

        if set_other_to_random:
            self.word_emb.weight[self.PAD_IDX].data.copy_(torch.tensor(np.random.randn(self.emb_dim)))
            self.word_emb.weight[self.START_IDX].data.copy_(torch.tensor(np.random.randn(self.emb_dim)))
            self.word_emb.weight[self.EOS_IDX].data.copy_(torch.tensor(np.random.randn(self.emb_dim)))

        if self.freeze_embeddings:
            self.emb_grad_mask = np.zeros(self.word_emb.weight.shape, dtype=np.float32)
            self.emb_grad_mask[self.UNK_IDX, :] = 1
            self.emb_grad_mask[self.PAD_IDX, :] = 1
            self.emb_grad_mask[self.START_IDX, :] = 1
            self.emb_grad_mask[self.EOS_IDX, :] = 1
            self.emb_grad_mask = to_gpu(torch.tensor(self.emb_grad_mask))
        else:
            self.emb_grad_mask = None

        if decode_with_embeddings:
            self.lm = nn.GRU(self.emb_dim, self.h_dim, dropout=rnn_dropout)
            self.lm_fc = nn.Linear( self.h_dim, self.emb_dim)
        else:
            self.lm = nn.GRU(self.emb_dim, self.h_dim, dropout=rnn_dropout)
            self.lm_fc = nn.Linear( self.h_dim, n_vocab)

        self.lm_params = filter(lambda p: p.requires_grad, self.lm.parameters())

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def decode(self, h):
        y = self.lm_fc(h)
        if self.decode_with_embeddings:
            y = cosine_sim_matrix(y, self.word_emb.weight)
        return y

    def mask_embedding_grad(self):
        if self.freeze_embeddings:
            self.word_emb.weight.grad *= self.emb_grad_mask

    def forward(self, x):
        x = to_gpu(x)
        inputs_emb = self.word_emb(x)
        outputs, _ = self.lm(inputs_emb, None)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len * mbsize, -1)
        y = self.decode(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        recon_loss = self.get_recon_loss(x, y)

        return {
            'recon_loss': recon_loss,
            'y': y
        }

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs) if self.training else inputs

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x (z_dim+c_dim)
        init_h = z.unsqueeze(0)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len * mbsize, -1)
        y = self.decode(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def get_recon_loss(self, sentence, y):
        mbsize = sentence.size(1)
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = to_gpu(pad_words)
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)
        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True, weight=self.weights
        )
        return recon_loss

    def sample_sentence(self, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb], 2)

            output, h = self.lm(emb, None)
            y = self.decode(output.view((1, -1))).view(-1)
            y = F.softmax(y / temp, dim=0)

            idx = torch.multinomial(y, num_samples=1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return to_gpu(outputs)
        else:
            return outputs

    def forward_multiple(self, x, n_times=16):
        x = to_gpu(x)
        ys = []
        recon_losses = []

        for i in range(x.size(1)):
            target = x[:, i:i + 1].repeat(1, n_times)
            y = self.forward(target)['y']
            recon_losses.append(self.get_recon_loss(target, y))
            ys.append(y)
        return {
            'recon_losses': recon_losses,
            'ys': ys
        }
