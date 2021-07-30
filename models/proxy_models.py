#linear projection
import numpy as np
import os
import time, collections, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from box import Box
from ..utility.noise import noisy
from model_utils import reparameterize, log_prob, loss_kl
from ..utility.utils import *
from ..utility.vocab import Vocab
from ..utility.batchify import *
import yaml
from ..utility.meter import AverageMeter
import argparse
from components import DiCLF, MultCLF, Decoder, Project, Encoder
from gradflow_check import *
from shutil import copyfile
from tensorboardX import SummaryWriter


class linAE(nn.Module):
    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()

        self.vocab = vocab
        self.args = args

        self.E_c = Encoder(vocab, args.content, embed=None, initrange=initrange)

        self.E_f = Encoder(vocab, args.form, embed=None, initrange=initrange)

        self.Gen = Decoder(self.vocab, self.args.decoder)

        self.Mot_f = DiCLF(self.args.form)
        self.Adv_f = DiCLF(self.args.form)

        self.Mot_c = MultCLF(self.args.content)
        self.Adv_c = MultCLF(self.args.content)

        self.params = list(self.E_c.params) + list(self.E_f.params) + list(self.Gen.params) + \
                 list(self.Mot_c.params) + list(self.Mot_f.params) #list(self.embed.parameters()) 

        self.opt = optim.Adam(params=filter(lambda p: p.requires_grad, self.params), lr=self.args.lr, betas=(0.5, 0.999))

    def reconstruct(self, inputs, seq_lens, targets, is_train=False):
       
        _, z_c = self.E_c(inputs, seq_lens)
        _, z_f  =self.E_f(inputs, seq_lens)
        logits = self.Gen(z_c, z_f, inputs, targets)
        return logits

    def forward(self, inputs, targets, seq_lens, f_labels, c_labels, is_train=False):

        _inputs = noisy(self.vocab, inputs, *self.args.noise) if is_train else inputs

        h_c, z_c = self.E_c(_inputs, seq_lens)
        h_f, z_f = self.E_f(_inputs, seq_lens)

        #form losses
        #mot
        if self.args.f_mot != 0:
            f_mot_op = self.Mot_f(z_f)
            f_mot_loss = self.Mot_f.loss(f_mot_op, f_labels)
        else:
            f_mot_loss = 0

        #adv 
        if self.args.f_adv != 0:
    
            f_adv_op = self.Adv_f(z_c.detach().clone())
            f_adv_loss = self.Adv_f.loss(f_adv_op, f_labels)
            f_adv_ent = self.Adv_f.loss_entropy(self.Adv_f(z_c))
        
        else:
            f_adv_loss = 0
            f_adv_ent = 0

        #content losses
        #mot
        if self.args.c_mot != 0:
            c_mot_op = self.Mot_c(z_c)
            c_mot_loss = self.Mot_c.loss(c_mot_op, c_labels)
        else:
            c_mot_loss = 0

        #adv 
        if self.args.c_adv != 0:

            c_adv_op = self.Adv_c(z_f.detach().clone())
            c_adv_loss = self.Adv_c.loss(c_adv_op, c_labels)
            c_adv_ent = self.Adv_c.loss_entropy(self.Adv_c(z_f))
        else:
            c_adv_loss = 0
            c_adv_ent = 0

        #reconstruction
        logits = self.Gen(z_c, z_f, inputs, targets)
        rec_loss = self.Gen.loss(logits, targets)
            
        losses = {
            'f_mot': f_mot_loss,
            'f_adv': f_adv_loss,
            'f_ent': f_adv_ent,
            'c_mot': c_mot_loss,
            'c_adv': c_adv_loss,
            'c_ent': c_adv_ent,
            'rec': rec_loss,
        }

        return losses

    def clf_forward(self, inputs, targets, seq_lens, f_labels, c_labels, is_train=False):
        _inputs = noisy(self.vocab, inputs, *self.args.noise) if is_train else inputs

        h_c, z_c = self.E_c(_inputs, seq_lens)
        h_f, z_f = self.E_f(_inputs, seq_lens)

        if self.args.f_adv != 0:

            adv_f = self.Adv_f(z_c.detach().clone())
            f_adv_loss = self.Adv_f.loss(adv_f, f_labels)
        
        else:
            f_adv_loss = 0

        if self.args.c_adv != 0:
            adv_c = self.Adv_c(z_f.detach().clone())
            c_adv_loss = self.Adv_c.loss(adv_c, c_labels)
        
        else:
            c_adv_loss = 0

        logits = self.Gen(z_c, z_f, inputs, targets)
        rec_loss = self.Gen.loss(logits, targets)

        losses = {
            'rec': rec_loss,
            'f_adv': f_adv_loss,
            'c_adv': c_adv_loss
        }

        return losses

    def clf_step(self, losses):
        self.set_zero_grad()
        losses['rec'].backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.params, self.args.max_grad_norm)
        self.opt.step()

        if self.args.c_adv != 0:
            self.Adv_c.step(losses['c_adv'])

        if self.args.f_adv != 0:
            # self.Adv_f.set_zero_grad()
            self.Adv_f.step(losses['f_adv'])
        
    def loss(self, losses, beta=None):
        if not beta:
            beta = self.args.beta

        ae_loss = self.args.l_rec * losses['rec'] + \
            self.args.f_mot * losses['f_mot'] - self.args.f_adv * losses['f_ent']  + \
                self.args.c_mot * losses['c_mot'] - self.args.c_adv * losses['c_ent'] 
            
        return {
            'ae_loss': ae_loss,
            'f_mot': losses['f_mot'],
            'f_adv': losses['f_adv'],
            'c_mot': losses['c_mot'],
            'c_adv': losses['c_adv']
        }

    def set_zero_grad(self):
        self.opt.zero_grad()

    def step(self, losses):

        self.set_zero_grad()
        losses['ae_loss'].backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.params, self.args.max_grad_norm)
        self.opt.step()

        if self.args.c_adv != 0:
            self.Adv_c.step(losses['c_adv'])

        if self.args.f_adv != 0:

            self.Adv_f.step(losses['f_adv'])