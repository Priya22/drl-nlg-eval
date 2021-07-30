from eval_utils import *
import pickle as pkl
import argparse
import time
import os
import random
import collections
import numpy as np
import torch
import yaml
from shutil import copyfile
from ..utility.utils import *
from ..utility.vocab import Vocab
from ..utility.batchify import *
import yaml
from yaml2object import YAMLObject

root_dir = 'results/nlg/' #saved models

def style_transfer(args):

    #change to desired config
    config_names = ['all']
    config_types = ['disae']

    res = {}
    for config, cname in zip(config_names, config_types):
        print(config)
        res[config] = {}
        
        config_name = config
        
        config_file = 'config/nlg_lin/config.yaml'
        if cname == 'baseline':
            config_file = 'config/nlg_lin/baseline.yaml'
            config_name = 'all'
        
        with open(config_file, 'r') as f:
            all_args = yaml.safe_load(f)

        args = get_args(all_args, config_name)
        args = Box(args)
        
        args.save_dir = os.path.join(root_dir, config)
        
        log_file = 'stdout' 
        vocab_file = os.path.join(args.save_dir, 'vocab.txt')

        # if not os.path.isfile(vocab_file):
        #     Vocab.build(train_sents, vocab_file, args.vocab.vocab_size)
        vocab = Vocab(vocab_file)
        logging('# vocab size {}'.format(vocab.size), log_file)

        set_seed(args.seed)

        cuda = args.cuda and torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        args.device = device
        args.type = 'para'

        vocab_file = os.path.join(args.save_dir, 'vocab.txt')

        # if not os.path.isfile(vocab_file):
        #     Vocab.build(train_sents, vocab_file, args.vocab.vocab_size)
        vocab = Vocab(vocab_file)
        logging('# vocab size {}'.format(vocab.size), log_file)

        set_seed(args.seed)

        cuda = args.cuda and torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        args.device = device

        test_batches1, test_batches2 = get_test_batches(args, vocab)

        if cname == 'disae':
            model = linAE(vocab, args.model)
        elif cname == 'para':
            model = pLinAE(vocab, args.model)
        elif cname == 'baseline':
            model = CondDecode(vocab, args.model)

        name = 'best_model.pt'
        print(args.save_dir)
        path = os.path.join(args.save_dir, name)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        model.to(args.device)
        model.eval()

        ip_sents = []

        for i in range(len(test_batches1)):
        #i = 0
            ip1, tg1, sl1, fl1, cl1 = test_batches1[i]
            ip2, tg2, sl2, fl2, cl2 = test_batches2[i]
            
            
            
            ip_sents1 = get_sents(ip1, vocab)
            ip_sents2 = get_sents(ip2, vocab)
        
            op1, op2 = transfer_style(ip1, ip2, sl1, sl2, fl1, fl2, model, cname)
            
            op_sents1 = get_outputs(op1, vocab)
            op_sents2 = get_outputs(op2, vocab)
            
            for ind in range(len(ip_sents1)):
                print("Input: ", " ".join(ip_sents1[ind]))
                print("Target: ", " ".join(ip_sents2[ind]))
                print("Output: ", " ".join(op_sents1[ind]))
                print()
                print("Input: ", " ".join(ip_sents2[ind]))
                print("Target: ", " ".join(ip_sents1[ind]))
                print("Output: ", " ".join(op_sents2[ind]))
                print()

def ft_classify(clf_model, test_batches1, test_batches2, res, config='baseline'):
    f = 'stdout'
    for i in range(len(test_batches1)):

        ip1, tg1, sl1, fl1, cl1 = test_batches1[i]
        ip2, tg2, sl2, fl2, cl2 = test_batches2[i]
        
        
        fl1 = fl1.detach().cpu().numpy()
        fl2 = fl2.detach().cpu().numpy()
        
        sents1 = res[config]['outputs'][i][1]
        sents2 = res[config]['outputs'][i][0]
        
        for l, s in zip(fl1, sents1):
            ft_str = " ".join(["__label__"+str(l), " ".join(s)])
            print(ft_str, file=f)
        for l, s in zip(fl2, sents2):
            ft_str = " ".join(["__label__"+str(l), " ".join(s)])
            print(ft_str, file=f)

    print(config)
    print_results(*clf_model.test('results/new_ent/nlg_lin/' + config + '/ft_'+config+'.txt'))
    print()

def get_bleus(res):
    #example: BLEU
    for name, results in res.items():
        sbleus = []
        c_sims = []
        f_sims = []
        for batch, _ in enumerate(results['inputs']):
            for ip, op in zip(results['inputs'][batch][0], results['outputs'][batch][1]):
                sbleus.append(get_self_bleu(op, ip))
            for ip, op in zip(results['inputs'][batch][1], results['outputs'][batch][0]):
                c_sims.append(get_self_bleu(op, ip))

            for ip, op in zip(results['inputs'][batch][0], results['outputs'][batch][0]):
                f_sims.append(form_sim(op, ip))
            for ip, op in zip(results['inputs'][batch][1], results['outputs'][batch][1]):
                f_sims.append(form_sim(op, ip))
                
        print(name, np.mean(sbleus))
        print(name, np.mean(c_sims))
        print(name, np.mean(f_sims))
