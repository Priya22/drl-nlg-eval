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
from proxy_models import linAE
from para_models import pLinAE
from gradflow_check import *
from shutil import copyfile
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--config', metavar='FILE', required=True, 
                    help='path to config file. ')
parser.add_argument('--config_name', type = str, required=True, 
                    help='name of configuration (see config file)')
parser.add_argument('--checkpoint_dir', metavar='FOLDER', required=False, 
                    help='path to checkpoint folder')

class disTrainer:
    def __init__(self, args, vocab, data):
        self.args = args
        
        self.log_file = os.path.join(args.save_dir, 'log.txt')

        tfboard_dir = os.path.join(self.args.save_dir, 'tf_board')

        if not os.path.isdir(tfboard_dir):
            os.mkdir(tfboard_dir)

        self.writer = SummaryWriter(tfboard_dir)

        self.train_batches = data['train_batches']
        self.valid_batches = data['valid_batches']
        #self.test_batches = data['test_batches']
        self.vocab = vocab 

        self.best_val_loss = None 
        self.epoch = 0
        self.pretrain_epoch = 0
        
        #set KL beta
        if self.args.warmup == -1:
            self.args.model.beta = 0.0
        if self.args.warmup == 0:
            self.args.model.beta = 1.0
        else:
            self.args.model.beta = 0.1

        self.model = linAE(vocab, args.model).to(args.device)

    def save_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = {
            'model': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'beta': self.model.args.beta,
            'epoch': self.epoch,
            'pretrain_epoch': self.pretrain_epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.epoch = ckpt['epoch'] + 1
        self.pretrain_epoch = ckpt['pretrain_epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.model.args.beta = ckpt['beta']

    def load_model(self, name):
        model = linAE(self.vocab, self.args.model)
        path = os.path.join(self.args.save_dir, name)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        model.to(self.args.device)
        return model


    def evaluate(self, model, batches):
        model.eval()
        with torch.no_grad():
            meters = collections.defaultdict(lambda: AverageMeter())
            with torch.no_grad():
                for inputs, targets, seq_lens, f_labels, c_labels in batches:
                    losses = model(inputs, targets, seq_lens, f_labels, c_labels, is_train=False)
                    comb_losses = model.loss(losses, beta=0.0)
                    all_losses = {**losses, **comb_losses}
                    for k, v in all_losses.items():
                        try:
                            meters[k].update(v.item(), inputs.size(1))
                        except:
                            pass

            avg = {k: meter.avg for k, meter in meters.items()}
            #loss = model.loss(avg)
            #meters['loss'].update(loss)
            #meters[]
            meters.update(avg)
            return meters

    def train_clf_epoch(self, epoch):
        start_time = time.time()
        logging("Time: " + str(start_time), self.log_file)

        self.model.train()

        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(self.train_batches)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            true_it = i + 1 + epoch * len(indices)

            inputs, targets, seq_lens, f_labels, c_labels = self.train_batches[idx]

            losses = self.model.clf_forward(inputs, targets, seq_lens, f_labels, c_labels, is_train=True)
            self.model.clf_step(losses)

            for k,v in losses.items():
                try:
                    meters[k].update(v.item())
                except:
                    pass
                    
            if (i+1) % self.args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    try:
                        log_output += ' {} {:.2f},'.format(k, meter.avg)
                        self.writer.add_scalar("pretrain/train/" + k, meter.avg, true_it)
                        meter.clear()
                    except:
                        pass
                log_output += ' {} {:.2f}'.format('beta', self.model.args.beta)
                logging(log_output, self.log_file)

        valid_meters = self.evaluate(self.model, self.valid_batches)

        logging('-' * 80, self.log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)

        for k, meter in valid_meters.items():
            try:
                log_output += ' {} {:.2f},'.format(k, meter)
                self.writer.add_scalar("pretrain/valid/" + k, meter, true_it)
            except:
                pass
        
        logging(log_output, self.log_file)
        logging("Saving checkpoint: ", self.log_file)

        self.save_checkpoint()

    def train_epoch(self, epoch):
        start_time = time.time()
        logging("Time: " + str(start_time), self.log_file)

        self.model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(self.train_batches)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            true_it = i + 1 + epoch * len(indices)

            inputs, targets, seq_lens, f_labels, c_labels = self.train_batches[idx]
            
            if self.args.warmup > 0:
                self.model.args.beta = min(1, self.model.args.beta + 1./(self.args.warmup*len(self.train_batches)))

            losses = self.model(inputs, targets, seq_lens, f_labels, c_labels, is_train=True)
            comb_losses = self.model.loss(losses)
            self.model.step(comb_losses)

            all_losses = {**losses, **comb_losses}
            for k, v in all_losses.items():
                try:
                    meters[k].update(v.item())
                except:
                    pass
            
            if (i+1) % self.args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    try:
                        log_output += ' {} {:.2f},'.format(k, meter.avg)
                        self.writer.add_scalar("train/train/" + k, meter.avg, true_it)
                        meter.clear()
                    except:
                        pass
                log_output += ' {} {:.2f}'.format('beta', self.model.args.beta)
                logging(log_output, self.log_file)
            
        # plot_grad_flow(self.model.named_parameters())
        valid_meters = self.evaluate(self.model, self.valid_batches)

        logging('-' * 80, self.log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            try:
                log_output += ' {} {:.2f},'.format(k, meter)
                self.writer.add_scalar("train/valid/" + k, meter, true_it)
            except:
                pass
        
        #logging(log_output, self.log_file)
        #log_output = ''
        if not self.best_val_loss or valid_meters['ae_loss'] < self.best_val_loss:
            log_output += ' | saving best model'
            ckpt = {'model': self.model.state_dict(), 'best_epoch': epoch}
            torch.save(ckpt, os.path.join(self.args.save_dir, 'best_model.pt'))
            self.best_val_loss = valid_meters['ae_loss']

        logging(log_output, self.log_file)
        logging("Saving checkpoint: ", self.log_file)

        self.save_checkpoint()
        #self.epoch += 1

    def train(self):
        
        #pretrain adversarial classifiers
        while self.pretrain_epoch < self.args.pretrain_epochs:
            logging("Adversarial pretraining: Epoch " + str(self.pretrain_epoch), self.log_file)
            self.train_clf_epoch(self.pretrain_epoch)
            self.pretrain_epoch += 1
        
        logging("Done pretraining.", self.log_file)

        while self.epoch < self.args.num_epochs:

            logging("Training epoch: " + str(self.epoch), self.log_file)
            #iter_ = self.epoch * len(self.train)
            if  (self.args.warmup != -1) and (self.epoch % self.args.cycle == 0):
                self.model.args.beta = 0.1
                logging("KL Annealing restart: ", self.log_file)

            self.train_epoch(self.epoch)
            self.epoch += 1
        
        logging("Done training: ", self.log_file)

class paraTrainer:
    def __init__(self, args, vocab, data):
        self.args = args
        
        self.log_file = os.path.join(args.save_dir, 'log.txt')

        tfboard_dir = os.path.join(self.args.save_dir, 'tf_board')

        if not os.path.isdir(tfboard_dir):
            os.mkdir(tfboard_dir)

        self.writer = SummaryWriter(tfboard_dir)

        self.train_batches = data['train_batches']
        self.valid_batches = data['valid_batches']
        #self.test_batches = data['test_batches']
        self.vocab = vocab 

        self.best_val_loss = None 
        self.epoch = 0
        self.pretrain_epoch = 0
        
        #set KL beta
        if self.args.warmup == -1:
            self.args.model.beta = 0.0
        if self.args.warmup == 0:
            self.args.model.beta = 1.0
        else:
            self.args.model.beta = 0.1

        self.model = pLinAE(vocab, args.model).to(args.device)

    def save_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = {
            'model': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'beta': self.model.args.beta,
            'epoch': self.epoch,
            'pretrain_epoch': self.pretrain_epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.epoch = ckpt['epoch'] + 1
        self.pretrain_epoch = ckpt['pretrain_epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.model.args.beta = ckpt['beta']

    def load_model(self, name):
        model = pLinAE(self.vocab, self.args.model)
        path = os.path.join(self.args.save_dir, name)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        model.to(self.args.device)
        return model



    def evaluate(self, model, batches):
        model.eval()
        with torch.no_grad():
            meters = collections.defaultdict(lambda: AverageMeter())
            with torch.no_grad():
                for inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2 in batches:
                    losses = model(inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2, is_train=False)
                    comb_losses = model.loss(losses, beta=0.0)
                    all_losses = {**losses, **comb_losses}
                    for k, v in all_losses.items():
                        try:
                            meters[k].update(v.item(), inputs1.size(1)*2)
                        except:
                            pass

            avg = {k: meter.avg for k, meter in meters.items()}
            #loss = model.loss(avg)
            #meters['loss'].update(loss)
            #meters[]
            meters.update(avg)
            return meters

    def train_clf_epoch(self, epoch):
        start_time = time.time()
        logging("Time: " + str(start_time), self.log_file)

        self.model.train()

        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(self.train_batches)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            true_it = i + 1 + epoch * len(indices)

            inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2 = self.train_batches[idx]

            losses = self.model.clf_forward(inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2, is_train=True)
            self.model.clf_step(losses)

            for k,v in losses.items():
                try:
                    meters[k].update(v.item())
                except:
                    pass
                    
            if (i+1) % self.args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    try:
                        log_output += ' {} {:.2f},'.format(k, meter.avg)
                        self.writer.add_scalar("pretrain/train/" + k, meter.avg, true_it)
                        meter.clear()
                    except:
                        pass
                log_output += ' {} {:.2f}'.format('beta', self.model.args.beta)
                logging(log_output, self.log_file)

        valid_meters = self.evaluate(self.model, self.valid_batches)

        logging('-' * 80, self.log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            try:
                log_output += ' {} {:.2f},'.format(k, meter)
                self.writer.add_scalar("pretrain/valid/" + k, meter, true_it)
            except:
                pass
        
        logging(log_output, self.log_file)
        logging("Saving checkpoint: ", self.log_file)
        

        self.save_checkpoint()

    def train_epoch(self, epoch):
        start_time = time.time()
        logging("Time: " + str(start_time), self.log_file)

        self.model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(self.train_batches)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            true_it = i + 1 + epoch * len(indices)

            inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2 = self.train_batches[idx]
            
            if self.args.warmup > 0:
                self.model.args.beta = min(1, self.model.args.beta + 1./(self.args.warmup*len(self.train_batches)*self.args.batch_size))

            losses = self.model(inputs1, targets1, seq_lens1, f_labels1, inputs2, targets2, seq_lens2, f_labels2, is_train=True)
            comb_losses = self.model.loss(losses)
            self.model.step(comb_losses)

            all_losses = {**losses, **comb_losses}
            for k, v in all_losses.items():
                try:
                    meters[k].update(v.item())
                except:
                    pass
            
            if (i+1) % self.args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    try:
                        log_output += ' {} {:.2f},'.format(k, meter.avg)
                        self.writer.add_scalar("train/train/" + k, meter.avg, true_it)
                        meter.clear()
                    except:
                        pass
                logging(log_output, self.log_file)
            
        #plot_grad_flow(self.model.named_parameters())
        valid_meters = self.evaluate(self.model, self.valid_batches)

        logging('-' * 80, self.log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            try:
                log_output += ' {} {:.2f},'.format(k, meter)
                self.writer.add_scalar("train/valid/" + k, meter, true_it)
            except:
                pass
        
        #logging(log_output, self.log_file)
        #log_output = ''
        if not self.best_val_loss or valid_meters['ae_loss'] < self.best_val_loss:
            log_output += ' | saving best model'
            ckpt = {'model': self.model.state_dict(), 'best_epoch': epoch}
            torch.save(ckpt, os.path.join(self.args.save_dir, 'best_model.pt'))
            self.best_val_loss = valid_meters['ae_loss']

        logging(log_output, self.log_file)
        logging("Saving checkpoint: ", self.log_file)

        self.save_checkpoint()
        #self.epoch += 1

    def train(self):

        while self.pretrain_epoch < self.args.pretrain_epochs:
            logging("Adversarial pretraining: Epoch " + str(self.pretrain_epoch), self.log_file)
            self.train_clf_epoch(self.pretrain_epoch)
            self.pretrain_epoch += 1
        
        logging("Done pretraining.", self.log_file)

        while self.epoch < self.args.num_epochs:

            logging("Training epoch: " + str(self.epoch), self.log_file)
            if (self.args.warmup != -1) and self.epoch % self.args.cycle == 0:
                self.model.args.beta = 0.1
                logging("KL Annealing restart: ", self.log_file)

            self.train_epoch(self.epoch)
            self.epoch += 1
        
        logging("Done training: ", self.log_file)


def main(args):
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    log_file = os.path.join(args.save_dir, 'log.txt')

    logging(str(args.to_dict()), log_file)

    data_folder = args.data 
    train_path = os.path.join(data_folder, 'train')
    valid_path = os.path.join(data_folder, 'valid')
    #test_path = os.path.join(data_folder, 'test')

    set_seed(args.seed)

    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    args.device = device

    if args.type == 'disae':
        train_sents, train_flabels, train_clabels = load_data(train_path)
        valid_sents, valid_flabels, valid_clabels = load_data(valid_path)

        logging('# train sents {}, tokens {}'.format(
            len(train_sents), sum(len(s) for s in train_sents)), log_file)

        logging('# valid sents {}, tokens {}'.format(
            len(valid_sents), sum(len(s) for s in valid_sents)), log_file)

        
        vocab_file = os.path.join(args.save_dir, 'vocab.txt')

        if not os.path.isfile(vocab_file):
            Vocab.build(train_sents, vocab_file, args.vocab.vocab_size)
        vocab = Vocab(vocab_file)
        logging('# vocab size {}'.format(vocab.size), log_file)


        batcher = MultLabelled()
        train_batches, _ = batcher.get_batches(train_sents, train_flabels, train_clabels, vocab, args.batch_size, args.device)
        valid_batches, _ = batcher.get_batches(valid_sents, valid_flabels, valid_clabels, vocab, args.batch_size, args.device)
        #test_batches, _ = batcher.get_batches(test_sents, test_flabels, test_clabels, vocab, args.batch_size, args.device)
        
        logging("Number of train batches: {}".format(len(train_batches)), log_file)
        logging("Number of valid batches: {}".format(len(valid_batches)), log_file)
        #logging("Number of test batches: {}".format(len(test_batches)), log_file)

        data = {
            'train_batches': train_batches,
            'valid_batches': valid_batches,
        }

        modelTrainer = disTrainer(args, vocab, data) 

        if args.resume:
            if os.path.isfile(os.path.join(args.save_dir, 'ckpt.pt')):
                modelTrainer.load_checkpoint()

        #train
        modelTrainer.train()

    elif args.type == 'para':
        train_sents1, train_sents2, train_flabels1, train_flabels2 = load_parallel(train_path)
        valid_sents1, valid_sents2, valid_flabels1, valid_flabels2 = load_parallel(valid_path)
        #test_sents1, test_sents2, test_flabels1, test_flabels2 = load_parallel(test_path)

        logging('# train sents {}, tokens {}'.format(
            len(train_sents1), sum(len(s) for s in train_sents1)), log_file)

        logging('# valid sents {}, tokens {}'.format(
            len(valid_sents1), sum(len(s) for s in valid_sents1)), log_file)
        # logging('# test sents {}, tokens {}'.format(
        #     len(test_sents1), sum(len(s) for s in test_sents1)), log_file)

        
        vocab_file = os.path.join(args.save_dir, 'vocab.txt')

        if not os.path.isfile(vocab_file):
            Vocab.build(train_sents1 + train_sents2, vocab_file, args.vocab.vocab_size)
        vocab = Vocab(vocab_file)
        logging('# vocab size {}'.format(vocab.size), log_file)

        batcher = Paraphrase()
        train_batches, _ = batcher.get_batches([train_sents1, train_sents2], [train_flabels1, train_flabels2], vocab, args.batch_size, args.device)
        valid_batches, _ = batcher.get_batches([valid_sents1, valid_sents2], [valid_flabels1, valid_flabels2], vocab, args.batch_size, args.device)
        #test_batches, _ = batcher.get_batches([test_sents1, test_sents2], [test_flabels1, test_flabels2], vocab, args.batch_size, args.device)
        
        logging("Number of train batches: {}".format(len(train_batches)), log_file)
        logging("Number of valid batches: {}".format(len(valid_batches)), log_file)
        #logging("Number of test batches: {}".format(len(test_batches)), log_file)

        data = {
            'train_batches': train_batches,
            'valid_batches': valid_batches,
            #'test_batches': test_batches
        }

        modelTrainer = paraTrainer(args, vocab, data) 

        if args.resume:
            if os.path.isfile(os.path.join(args.save_dir, 'ckpt.pt')):
                modelTrainer.load_checkpoint()

        #train
        modelTrainer.train()

    else:
        logging("Invalid type: exiting.")


def get_args(all_args, name):
    for key, val in all_args[name]['common'].items():
        all_args['common'][key] = val

    for key, val in all_args[name]['model'].items():
        all_args['common']['model'][key] = val
    
    return all_args['common']

if __name__=='__main__':
    parse_args = parser.parse_args()

    config_file = parse_args.config
    config_name = parse_args.config_name
    chkpt_dir = parse_args.checkpoint_dir

    with open(config_file, 'r') as f:
        all_args = yaml.safe_load(f)

    args = get_args(all_args, config_name)
    args = Box(args)

    if chkpt_dir is not None:
        args.save_dir = chkpt_dir

    #copy config file
    copyfile(config_file, os.path.join(args.save_dir, 'config.yaml'))
    args.model.noise = [float(x) for x in args.model.noise.split(",")]

    main(args)