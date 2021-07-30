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

class pLinAE(nn.Module):
    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()

        self.vocab = vocab
        self.args = args

        self.E_c = Encoder(vocab, args.content, embed=None, initrange=initrange)

        self.E_f = Encoder(vocab, args.form, embed=None, initrange=initrange)

        self.Gen = Decoder(vocab, self.args.decoder)

        self.Mot_f = DiCLF(self.args.form)
        self.Adv_f = DiCLF(self.args.form)

        self.params =  list(self.E_c.params) + list(self.E_f.params) + list(self.Gen.params) + \
            list(self.Mot_f.params) #+ list(self.embed.parameters())

        self.opt = optim.Adam(params=filter(lambda p: p.requires_grad, self.params), \
            lr=self.args.lr, betas=(0.5, 0.999))



    def forward(self, inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2, is_train=False):
        _inputs1 = noisy(self.vocab, inputs1, *self.args.noise) if is_train else inputs1
        _inputs2 = noisy(self.vocab, inputs2, *self.args.noise) if is_train else inputs2 

        _, z_c1 = self.E_c(_inputs1, seq_lens1)
        _, z_c2 = self.E_c(_inputs2, seq_lens2)
        _, z_f1 = self.E_f(_inputs1, seq_lens1)
        _, z_f2 = self.E_f(_inputs2, seq_lens2)


        #form losses
        #mot
        if self.args.f_mot != 0:
            f_mot_op1 = self.Mot_f(z_f1) 
            f_mot_op2 = self.Mot_f(z_f2) 
            f_mot_loss = self.Mot_f.loss(f_mot_op1, f_labels1) + self.Mot_f.loss(f_mot_op2, f_labels2)
        else:
            f_mot_loss = 0

        #adv 
        if self.args.f_adv != 0:

            f_adv_op1 = self.Adv_f(z_c1.detach().clone())
            f_adv_op2 = self.Adv_f(z_c2.detach().clone())
            f_adv_loss = self.Adv_f.loss(f_adv_op1, f_labels1) + self.Adv_f.loss(f_adv_op2, f_labels2)
        
            f_adv_ent = self.Adv_f.loss_entropy(self.Adv_f(z_c1)) + self.Adv_f.loss_entropy(self.Adv_f(z_c2))
        else:
            f_adv_loss = 0
            f_adv_ent = 0

        #paraphrase
        if self.args.c_para != 0:
            logits12 = self.Gen(z_c2, z_f1, inputs1, targets1)
            rec_loss12 = self.Gen.loss(logits12, targets1)
            logits21 = self.Gen(z_c1, z_f2, inputs2, targets2)
            rec_loss21 = self.Gen.loss(logits21, targets2)

            para_loss = rec_loss12 + rec_loss21
        else:
            para_loss = 0

        #reconstruction
        logits1 = self.Gen(z_c1, z_f1, inputs1, targets1)
        rec_loss1 = self.Gen.loss(logits1, targets1)
        logits2 = self.Gen(z_c2, z_f2, inputs2, targets2)
        rec_loss2 = self.Gen.loss(logits2, targets2)

        rec_loss = rec_loss1 + rec_loss2
     
        losses = {
            'f_mot': f_mot_loss,
            'f_adv': f_adv_loss,
            'f_ent': f_adv_ent,
            'c_para': para_loss,
            'rec': rec_loss,

        }

        return losses

    def clf_forward(self, inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2, is_train=False):
        _inputs1 = noisy(self.vocab, inputs1, *self.args.noise) if is_train else inputs1
        _inputs2 = noisy(self.vocab, inputs2, *self.args.noise) if is_train else inputs2 

        _, z_c1 = self.E_c(_inputs1, seq_lens1)
        _, z_c2 = self.E_c(_inputs2, seq_lens2)
        _, z_f1 = self.E_f(_inputs1, seq_lens1)
        _, z_f2 = self.E_f(_inputs2, seq_lens2)

        if self.args.f_adv != 0:

            adv_f1 = self.Adv_f(z_c1.detach().clone())
            adv_f2 = self.Adv_f(z_c2.detach().clone())
            f_adv_loss = self.Adv_f.loss(adv_f1, f_labels1) + self.Adv_f.loss(adv_f2, f_labels2)

        else:
            f_adv_loss = 0

        #reconstruction
        logits1 = self.Gen(z_c1, z_f1, inputs1, targets1)
        rec_loss1 = self.Gen.loss(logits1, targets1)
        logits2 = self.Gen(z_c2, z_f2, inputs2, targets2)
        rec_loss2 = self.Gen.loss(logits2, targets2)

        rec_loss = rec_loss1 + rec_loss2

        losses = {
            'rec': rec_loss,
            'f_adv': f_adv_loss
        }

        return losses

    def clf_step(self, losses):

        self.set_zero_grad()
        losses['rec'].backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.params, self.args.max_grad_norm)
        self.opt.step()

        if self.args.f_adv != 0:

            self.Adv_f.step(losses['f_adv'])



    def loss(self, losses, beta=None):
        if not beta:
            beta = self.args.beta

        ae_loss = self.args.l_rec * losses['rec']  + \
            self.args.f_mot * losses['f_mot'] - self.args.f_adv * losses['f_ent'] +\
                self.args.c_para * losses['c_para'] 

        return {
            'ae_loss': ae_loss,
            'f_mot': losses['f_mot'],
            'f_adv': losses['f_adv'],
            'c_para': losses['c_para']
        }

    def set_zero_grad(self):
        self.opt.zero_grad()

    def step(self, losses):

        self.set_zero_grad()
        losses['ae_loss'].backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.params, self.args.max_grad_norm)
        self.opt.step()

        if self.args.f_adv != 0:
            self.Adv_f.step(losses['f_adv'])
