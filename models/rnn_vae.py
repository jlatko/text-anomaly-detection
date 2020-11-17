import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain

from fast_transformers.builders import TransformerEncoderBuilder

builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=8,
    n_heads=8,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=1024
)

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


class RNN_VAE(nn.Module):
    """
    1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015).
    3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, n_vocab, h_dim=128, z_dim=128, p_word_dropout=0.3,
                 unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15,
                 use_input_embeddings=True, set_other_to_random=False,
                 set_unk_to_random=True, decode_with_embeddings=True,
                 rnn_dropout=0.3, mask_pad=True,
                 pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.PAD_IDX = pad_idx
        self.UNK_IDX = unk_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len
        self.freeze_embeddings = freeze_embeddings

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.p_word_dropout = p_word_dropout
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

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        # self.encoder = nn.GRU(self.emb_dim, h_dim, bidirectional=True)

        self.encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=12,
            n_heads=12,
            query_dimensions=64,
            value_dimensions=64,
            feed_forward_dimensions=self.emb_dim,
            attention_type="full",  # change this to use another
            # attention implementation
            activation="gelu"
        ).get()

        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        Decoder is GRU with `z` appended at its inputs
        """
        if decode_with_embeddings:
            self.decoder = nn.GRU(self.emb_dim + z_dim, z_dim, dropout=rnn_dropout)
            self.decoder_fc = nn.Linear(z_dim, self.emb_dim)

        else:
            self.decoder = nn.GRU(self.emb_dim + z_dim, z_dim, dropout=rnn_dropout)
            self.decoder_fc = nn.Linear(z_dim, n_vocab)

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.q_mu.parameters(),
            self.q_logvar.parameters()
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        )

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_params, self.decoder_params
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def decode(self, h):
        y = self.decoder_fc(h)
        if self.decode_with_embeddings:
            y = cosine_sim_matrix(y, self.word_emb.weight)
        return y

    def mask_embedding_grad(self):
        if self.freeze_embeddings:
            self.word_emb.weight.grad *= self.emb_grad_mask

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = to_gpu(eps)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        return to_gpu(z)

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

    def forward(self, sentence):
        """
        Params:
        -------
        sentence: sequence of word indices.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        sentence = to_gpu(sentence)
        mbsize = sentence.size(1)

        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = to_gpu(pad_words)

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z)

        # TODO: mask out <pad>
        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True, weight=self.weights
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1))

        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'y': y
        }

    def generate_sentences(self, batch_size):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []
        cs = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            samples.append(self.sample_sentence(z, raw=True))

        X_gen = torch.cat(samples, dim=0)

        return X_gen

    def sample_sentence(self, z, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z = z.view(1, 1, -1)
        h = z

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z], 2)

            output, h = self.decoder(emb, h)
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

    def generate_soft_embed(self, mbsize, temp=1):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        samples = []
        targets_c = []
        targets_z = []

        for _ in range(mbsize):
            z = self.sample_z_prior(1)

            samples.append(self.sample_soft_embed(z, temp=1))
            targets_z.append(z)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)

        return X_gen, targets_z

    def sample_soft_embed(self, z, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        z = z.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z], 2)

        h = torch.cat([z], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.word_emb(word).view(1, -1)]

        for i in range(self.MAX_SENT_LEN):
            output, h = self.decoder(emb, h)
            o = self.decoder(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                .astype('bool')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
