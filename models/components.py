import numpy as np
import os
import time, collections, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import yaml
import argparse
from gradflow_check import *
from shutil import copyfile
from tensorboardX import SummaryWriter

class Project(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.params = list(self.parameters())

    def encode(self, h):
        return self.h2mu(h), self.h2logvar(h)

    def forward(self, h):
        mu, logvar = self.encode(h)
        z = reparameterize(mu, logvar)

        return mu, logvar, z 

    def loss_kl(self, mu, logvar):
        return loss_kl(mu, logvar)



class Encoder(nn.Module):
    def __init__(self, vocab, args, embed=None, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.drop = nn.Dropout(self.args.dropout)
        self.embed = embed    
        if self.embed is None:
            self.embed = nn.Embedding(vocab.size, self.args.dim_emb)

        self.embed.weight.data.uniform_(-initrange, initrange)

        self.E = nn.LSTM(self.args.dim_emb, self.args.dim_h, self.args.nlayers,
                dropout=self.args.dropout if self.args.nlayers > 1 else 0, bidirectional=True)
        self.h2z = nn.Linear(args.dim_h*2, args.dim_z)

        self.params = list(self.E.parameters())

    def flatten(self):
        self.E.flatten_parameters()

    def forward(self, input, seq_lens):
        input = self.drop(self.embed(input))
        packed_vecs = nn.utils.rnn.pack_padded_sequence(input, seq_lens, enforce_sorted=False)
        packed_outputs, (hidden,_) = self.E(packed_vecs)
        # _, (h, _) = self.E(input)
        h = torch.cat([hidden[-2], hidden[-1]], 1)
        z = self.h2z(h)
        return h, z

class DiCLF(nn.Module):
    '''
    classifier used as discriminator. 
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epsilon = 1e-07
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.LeakyReLU(),
            nn.Linear(args.dim_d, args.n_classes))
        self.params = list(self.D.parameters())
        self.opt = optim.RMSprop(self.params, lr=args.lr)
        self.criterion = F.cross_entropy

    def forward(self, z):
        return self.D(z)
    
    def get_class(self, z, ind2class=None):
        logits = self(z)
        max_args = torch.argmax(logits, dim=1)

        if ind2class:
            assert logits.shape[-1] == len(ind2class), 'Size mismatch in number of classes!'
            return [ind2class[m] for m in max_args]

        else:
            return max_args
    
    def loss(self, op, labels):
        return self.criterion(op, labels)
    
    def loss_entropy(self, op):
        probs = F.softmax(op, dim=1)
        entropy = probs * torch.log(probs + self.epsilon)
        entropy = -1.0 * entropy.sum(dim=1)

        return entropy.mean()


        # b = F.softmax(op, dim=1) * F.log_softmax(op, dim=1)
        # b = -1.0 * b.sum(dim=1)
        # # probs = torch.sigmoid(op)
        # # b = probs * torch.log(probs + epsilon) + (1 - probs) * torch.log(1 - probs + epsilon)
        # return b.mean()

    def set_zero_grad(self):
        self.opt.zero_grad()

    def step(self, loss, retain_graph=False):
        #self.optD.zero_grad()
        self.set_zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
        self.opt.step()
        # self.zero_grad()

class MultCLF(nn.Module):
    '''
    classifier used as discriminator. 
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epsilon = 1e-07
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.LeakyReLU(),
            nn.Linear(args.dim_d, args.n_labels))
        self.criterion = nn.BCEWithLogitsLoss()
        self.params = list(self.D.parameters())
        self.opt = optim.RMSprop(self.params, lr=args.lr)

    def forward(self, z):
        return self.D(z)

    def get_class(self, logits, ind2class=None):
        #logits = self(z)
        #logits = nn.Sigmoid()(logits)
        pos = torch.where(logits>0.0, torch.Tensor([1]).to(self.args.device), torch.Tensor([0]).to(self.args.device))
        if ind2class:
            assert logits.shape[-1] == len(ind2class), 'Size mismatch in number of classes!'
            return [[ind2class[x] for x in m] for m in pos]

        else:
            return pos
    
    def loss(self, op, labels):
        return self.criterion(op, labels.t().type_as(op))
    
    def loss_entropy(self, op):
        probs = F.softmax(op, dim=1)
        entropy = probs * torch.log(probs + self.epsilon)
        entropy = -1.0 * entropy.sum(dim=1)

        return entropy.mean()
        
        # b = F.softmax(op, dim=1) * F.log_softmax(op, dim=1)
        # b = -1.0 * b.sum(dim=1)
        # return b.mean()

    def set_zero_grad(self):
        self.opt.zero_grad()
    
    def step(self, loss, retain_graph=False):
        #self.optD.zero_grad()
        self.set_zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
        self.opt.step()


class RegClf(nn.Module):
    """Adversarial Regularizer"""

    def __init__(self, args):
        super().__init__()
        
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.params = list(self.parameters())
        self.opt = optim.Adam(self.params, lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def forward(self, z):
        return self.loss_adv(z)

    def set_zero_grad(self):
        self.opt.set_zero_grad()

    def step(self, loss, retain_graph=False):

        self.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.opt.step()

class Decoder(nn.Module):
    def __init__(self, vocab, args, initrange=0.1):

        super().__init__()
        self.vocab = vocab
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.embed = nn.Embedding(self.vocab.size, self.args.dim_emb)
        self.embed.weight.data.uniform_(-initrange, initrange)
        #self.embed = embed
        self.proj = nn.Linear(args.dim_h, vocab.size)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)
        #to emb
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)

        #generator
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)

        self.params = list(self.parameters()) 
        
    def flatten(self):
        self.G.flatten_parameters()
    
    def decode_old(self, z_c, z_f, input, hidden=None):
        z = torch.cat([z_f, z_c], 1)
        #input = self.drop(self.embed(input)) + self.z2emb(z)
        sl = input.shape[0]
        z = z.unsqueeze(0).expand(sl, -1, -1)
        input_ = self.z2emb(z)
        output, hidden = self.G(input_, hidden)
        #output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def decode_step(self, ip, z, hidden=None):
        ip_ = self.drop(self.embed(ip)) + self.z2emb(z)
        ip_ = ip_.unsqueeze(0)
        o, (h,c) = self.G(ip_, hidden)
        o = self.drop(o)
        return o, (h,c)


    def decode(self, z_c, z_f, input, target, hidden=None, tf_ratio=None):

        if tf_ratio is None:
            tf_ratio = self.args.tf_ratio

        sl, bs = input.size()

        z = torch.cat([z_f, z_c], 1)
        ip = input[0,:] #sos 
        output_logits = torch.zeros((sl, bs, self.vocab.size), device=z.device)
        #hidden_states = torch.zeros((sl, bs, self.vocab.size)).type_as(input)

        for step in range(sl):
            o, hidden = self.decode_step(ip, z, hidden)

            logit = self.proj(o.view(-1, o.size(-1)))
            output_logits[step] = logit 

            ip = logit.argmax(dim=-1)
            teacher_force = random.random() < tf_ratio
            if teacher_force:
                ip = target[step,:]

        return output_logits

    def forward(self, z_c, z_f, input, target):
        #TODO: fix 
        #seems right
        #input = torch.zeros(1, len(z_c), dtype=torch.long, device=z_c.device).fill_(self.vocab.go)
        logits = self.decode(z_c, z_f, input, target)
        return logits

    def loss(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size()) #make it reduction None, and manually reduce for each sentence 
        return loss.sum(dim=0).mean()


    def generate(self, z_c, z_f, max_len, alg='top5'):
        bs = len(z_c)
        z = torch.cat([z_f, z_c], 1)
        #add beam search
        assert alg == 'greedy' or alg == 'sample' or alg == 'top5'
        sents = []
        input = torch.zeros(bs, dtype=torch.long, device=z_c.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            output, hidden = self.decode_step(input, z, hidden=hidden)
            logits = self.proj(output.view(-1, output.size(-1)))
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg=='sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t().view(-1)
            else:
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=-1,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[torch.arange(logits_exp.size(0)).unsqueeze(1), not_top5_indices] = 0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t().view(-1)
        #return #torch.cat(sents)
        return torch.stack(sents)