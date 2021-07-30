import argparse
import time
import os, sys
import random
import collections
import numpy as np
import torch
import yaml
from shutil import copyfile
from box import Box 
import pickle as pkl
from ..models.components import *

from ..models.proxy_models import linAE 
from ..models.para_models import pLinAE
from ..models.baseline import CondDecode

from ..utility.utils import *
from ..utility.vocab import Vocab
from ..utility.batchify import *
import yaml
from yaml2object import YAMLObject
from ..utility.meter import AverageMeter
from sklearn import manifold
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from eval_utils import get_vectors

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--config', metavar='FILE', required=True, 
                    help='path to config file. ')
parser.add_argument('--config_name', type = str, required=False, 
                    help='name of configuration (see config file)')
parser.add_argument('--checkpoint_dir', metavar='FOLDER', required=False, 
                    help='path to checkpoint folder')
parser.add_argument('--data', metavar='FOLDER', required=False, 
                    help='path to data folder')
parser.add_argument('--clf_aspect', type=str, default='form', help="what to classify for:")
parser.add_argument('--clf_input', type=str, default='form', help="input vector aspect:")
parser.add_argument('--batch_size', type=int, default=32, help="batch size:")
parser.add_argument('--num_epochs', type=int, default=5, help="number of epochs:")

def similarity(cl1, cl2):
    score = 0
    for v1, v2 in zip(cl1, cl2):
        if v1 == v2:
            score += 1
    
    return score/len(cl1)


def get_clf_metrics(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average='macro')

def get_multilabel_score(y_true, y_pred):
    scores = []
    for col in range(y_true.shape[-1]):
        true = y_true[:, col]
        pred = y_pred[:, col]
        p,r,f1,_ = get_clf_metrics(true, pred)
        scores.append([p, r, f1])
    scores = np.array(scores)
    scores = np.mean(scores, axis=0)

    return scores

def get_metric_dict(keys, vals):
    d = {}
    for k, v in zip(keys, vals):
        d[k] = v
    return d


class ClfTrainer:
    def __init__(self, args, vocab, data, aspect='form', ip='form', base='disae'):
        self.args = args
        self.aspect = aspect
        self.ip = ip
        self.log_file = os.path.join(args.save_dir, 'clf_eval_log.txt')
        self.base_name = base
        self.train_batches = data['train_batches']
        self.valid_batches = data['valid_batches']
        self.test_batches = data['test_batches']

        self.best_val_loss = None 
        self.epoch = 0

        if self.ip == 'form':
            args.model.form.dim_clf_z = args.model.form.dim_z 
        else:
            args.model.form.dim_clf_z = args.model.content.dim_z

        self.model = DiCLF(args.model.form).to(args.device)

        if self.aspect == 'content':
            if self.ip == 'form':
                args.model.content.dim_clf_z = args.model.form.dim_z 
            else:
                args.model.content.dim_clf_z = args.model.content.dim_z
            self.model = MultCLF(args.model.content).to(args.device)

        if base == 'disae':
            self.repModel = linAE(vocab, args.model)
            self.base_name = 'disae'
        elif base == 'para':
            self.repModel = pLinAE(vocab, args.model)
            self.base_name = 'para'

        elif base == 'baseline':
            self.repModel = CondDecode(vocab, args.model)
            self.base_name = 'baseline'
        
        #load base model 
        self.repModel = self.load_base(self.repModel)
        self.repModel.eval()

    def save_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = {
            'model': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self):
        path = os.path.join(self.args.save_dir, 'ckpt.pt')
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']

    def load_model(self,name):
        model = DiCLF(self.args.model.form)
        if self.aspect == 'content':
            model = MultCLF(self.args.model.content)
        path = os.path.join(self.args.save_dir, name)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        model.to(self.args.device)
        return model

    def load_base(self, m, name='ckpt.pt'):
        #base_dir = os.path.join(self.args.save_dir, '../')
        path = os.path.join(self.args.save_dir, '../', name)
        ckpt = torch.load(path)
        m.load_state_dict(ckpt['model'])
        m.to(self.args.device)
        m.eval()
        return m

    def get_base_outputs(self, inputs, seq_lens, f_labels):
        with torch.no_grad():
            z_c, z_f = get_vectors(inputs, seq_lens, f_labels, self.repModel, self.base_name)

            if self.ip == 'form':
                return z_f 
            else:
                return z_c 

        #return z_c, z_f

    def evaluate(self, model, batches):
        model.eval()
        with torch.no_grad():
            meters = collections.defaultdict(lambda: AverageMeter())
            #with torch.no_grad():
            for inputs, targets, seq_lens, flabels, clabels in batches:
                z = self.get_base_outputs(inputs, seq_lens, flabels)
                op = self.model(z)
                
                if self.aspect == 'form':
                    loss = self.model.loss(op, flabels)
                    y_true = flabels.detach().cpu().numpy()
                    y_pred = np.argmax(op.cpu().numpy(), axis=1)
                    p,r,f1,_ = get_clf_metrics(y_true, y_pred)
                else:
                    loss = self.model.loss(op, clabels)
                    y_true = clabels.t().detach().cpu().numpy()
                    y_pred = self.model.get_class(op).detach().cpu().numpy()
                    p, r, f1 = get_multilabel_score(y_true, y_pred)

                all_losses = {'loss': loss, 'precision': p, 'recall': r, 'f1': f1}
                for k, v in all_losses.items():
                    try:
                        meters[k].update(v.item(), inputs.size(1))
                    except:
                        pass

            avg = {k: meter.avg for k, meter in meters.items()}
            print(avg)
            #loss = model.loss(avg)
            #meters['loss'].update(loss)
            #meters[]
            meters.update(avg)
            return meters

    def train_epoch(self, epoch):
        start_time = time.time()
        logging("Time: " + str(start_time), self.log_file)

        self.model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(self.train_batches)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            inputs, targets, seq_lens, flabels, clabels = self.train_batches[idx]
            z = self.get_base_outputs(inputs, seq_lens, flabels)
            op = self.model(z)
            if self.aspect == 'form':
                loss = self.model.loss(op, flabels)
            elif self.aspect == 'content':
                loss = self.model.loss(op, clabels)
            else:
                sys.exit("Invalid aspect")

            self.model.opt.zero_grad()
            loss.backward()
            self.model.opt.step()

            all_losses = {'loss': loss}
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
                        meter.clear()
                    except:
                        logging("Exception error: ")
                        pass
                logging(log_output, self.log_file)
            

        valid_meters = self.evaluate(self.model, self.valid_batches)

        logging('-' * 80, self.log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            try:
                log_output += ' {} {:.2f},'.format(k, meter)
            except:
                pass
        
        #logging(log_output, self.log_file)
        #log_output = ''
        if not self.best_val_loss or valid_meters['loss'] < self.best_val_loss:
            log_output += ' | saving best model'
            ckpt = {'model': self.model.state_dict(), 'best_epoch': epoch}
            torch.save(ckpt, os.path.join(self.args.save_dir, 'best_model.pt'))
            self.best_val_loss = valid_meters['loss']

        logging(log_output, self.log_file)
        logging("Saving checkpoint: ", self.log_file)

        self.save_checkpoint()
        #self.epoch += 1

    def train(self):

        while self.epoch < self.args.num_epochs:

            logging("Training epoch: " + str(self.epoch), self.log_file)
            self.train_epoch(self.epoch)
            self.epoch += 1
        
        logging("Done training: ", self.log_file)

    def test(self):
        #load best model
        model = self.load_model('best_model.pt')
        test_meters = self.evaluate(model, self.test_batches)
        log_output = 'Test Performance: '
        for k, meter in test_meters.items():
           log_output += ' {} {:.2f},'.format(k, meter)
        logging(log_output, self.log_file)
        return {k: v for k,v in test_meters.items()}
          




################################################################
def runTrainer(args):
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    log_file = os.path.join(args.save_dir, 'log.txt')

    logging(str(args.to_dict()), log_file)

    data_folder = args.data 
    train_path = os.path.join(data_folder, 'train')
    valid_path = os.path.join(data_folder, 'valid')
    test_path = os.path.join(data_folder, 'test')

    train_sents, train_flabels, train_clabels = load_data(train_path)
    valid_sents, valid_flabels, valid_clabels = load_data(valid_path)
    test_sents, test_flabels, test_clabels = load_data(test_path)
    if args.aspect == 'form':
        train_labels, valid_labels, test_labels = train_flabels, valid_flabels, test_flabels
    else:
        train_labels, valid_labels, test_labels = train_clabels, valid_clabels, test_clabels

    
    logging('# train sents {}, tokens {}'.format(
        len(train_labels), sum(len(s) for s in train_sents)), log_file)

    logging('# valid sents {}, tokens {}'.format(
        len(valid_labels), sum(len(s) for s in valid_sents)), log_file)
    logging('# test sents {}, tokens {}'.format(
        len(test_labels), sum(len(s) for s in test_sents)), log_file)

    
    vocab_file = os.path.join(args.save_dir, '../', 'vocab.txt')

    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)

    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    args.device = device

    batcher = MultLabelled()
    train_batches, _ = batcher.get_batches(train_sents, train_flabels, train_clabels, vocab, args.batch_size, args.device)
    valid_batches, _ = batcher.get_batches(valid_sents, valid_flabels, valid_clabels, vocab, args.batch_size, args.device)
    test_batches, _ = batcher.get_batches(test_sents, test_flabels, test_clabels, vocab, args.batch_size, args.device)
    
    logging("Number of train batches: {}".format(len(train_batches)), log_file)
    logging("Number of valid batches: {}".format(len(valid_batches)), log_file)
    logging("Number of test batches: {}".format(len(test_batches)), log_file)

    data = {
        'train_batches': train_batches,
        'valid_batches': valid_batches,
        'test_batches': test_batches
    }

    modelTrainer = ClfTrainer(args, vocab, data, aspect=args.aspect, ip=args.ip, base=args.base) 

    if args.resume:
        if os.path.isfile(os.path.join(args.save_dir, 'ckpt.pt')):
            modelTrainer.load_checkpoint()

    #train
    modelTrainer.train()

    test_res = modelTrainer.test()
    return test_res
    
def get_args(all_args, name):
    for key, val in all_args[name]['common'].items():
        all_args['common'][key] = val

    for key, val in all_args[name]['model'].items():
        all_args['common']['model'][key] = val
    
    return all_args['common']


def run(parse_args, cname):

    config_file = parse_args.config
    config_name = parse_args.config_name
    chkpt_dir = parse_args.checkpoint_dir
    aspect = parse_args.clf_aspect
    ip = parse_args.clf_input
    num_epochs = parse_args.num_epochs
    data_folder = parse_args.data

    with open(config_file, 'r') as f:
        all_args = yaml.safe_load(f)

   
    args = get_args(all_args, config_name)
    args = Box(args)

    if chkpt_dir is not None:
        args.save_dir = chkpt_dir

    args.base = cname

    args.aspect = aspect 
    args.ip = ip 
    task_name = aspect + "_from_" + ip
    save_dir = os.path.join(args.save_dir, task_name)
    args.save_dir = save_dir
    args.num_epochs = num_epochs
   
    args.batch_size = parse_args.batch_size
    args.data = data_folder
    args.resume = 1
   

    test_res = runTrainer(args)

    return test_res


def main():

    main_args = parser.parse_args()
    print("Main args: ", main_args)
    batch_size = main_args.batch_size
    num_epochs = main_args.num_epochs
    root_dir = main_args.checkpoint_dir
    config_file = main_args.config
    data_folder = main_args.data

    aspects = ['form', 'content']
    inputs = ['form', 'content']

    save_root = root_dir
    config_names = ['ae', 'all', 'content', 'form', 'mot', 'para', 'para_f', 'baseline']
    config_types = ['disae', 'disae', 'disae', 'disae', 'disae', 'para', 'para', 'baseline']
    # config_names = ['baseline']
    # config_types = ['baseline']
   
    
    print(config_names)

    results = {}
    for config, cname in zip(config_names, config_types):
        results[config] = {}
        save_dir = os.path.join(save_root, config)
        for aspect in aspects:
            #results[config]['aspect'] = aspect
            for ip in inputs:
                results[config][aspect + '_from_' + ip] = {}
                
                arg_str = ' --config ' + config_file + ' --config_name ' + config + ' --checkpoint_dir ' + save_dir + \
    ' --clf_aspect ' + aspect + ' --clf_input ' + ip + ' --num_epochs ' + str(num_epochs) + ' --batch_size ' + str(batch_size) + ' --data ' + data_folder
                
                if cname == 'baseline':
                    config_root = os.path.split(os.path.normpath(config_file))
                    config_file = os.path.join(config_root[0], 'baseline.yaml')
                    arg_str = ' --config ' + config_file + ' --config_name all' + ' --checkpoint_dir ' + save_dir + \
    ' --clf_aspect ' + aspect + ' --clf_input ' + ip + ' --num_epochs ' + str(num_epochs) + ' --batch_size ' + str(batch_size) + ' --data ' + data_folder
                
                parse_args = parser.parse_args(arg_str.split())

                res = run(parse_args, cname)
                results[config][aspect + '_from_' + ip]['test_res'] = res
            print(root_dir)
            pkl.dump(results, open(os.path.join(root_dir, 'clf_eval_results_baseline.dict.pkl'), 'wb'))

if __name__=='__main__':
    main()