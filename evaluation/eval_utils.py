import argparse
import time
import os, sys, csv
import random
import collections
import numpy as np
import torch
import yaml
from shutil import copyfile
from nltk.translate import bleu_score
from box import Box
from ..models.components import *
#from para_models import paraAE
from ..models.proxy_models import linAE
from ..models.para_models import pLinAE
from ..models.trainers import get_args
from ..models.baseline import CondDecode
from ..utility.utils import *
from ..utility.vocab import Vocab
from ..utility.batchify import *
import yaml
from yaml2object import YAMLObject
from ..utility.noise import noisy
from sklearn import manifold
from collections import defaultdict
import matplotlib.pyplot as plt
from knlm import KneserNey
import fasttext

root_dir = 'results/nlg/' #saved models
data_folder = 'data/personage-nlg/labelled/'

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--config', metavar='FILE', required=True, 
                    help='path to config file. ')
parser.add_argument('--config_name', type = str, required=True, 
                    help='name of configuration (see config file)')
parser.add_argument('--checkpoint_dir', metavar='FOLDER', required=False, 
                    help='path to checkpoint folder')
parser.add_argument('--data_folder', metavar='FOLDER', required=False, 
                    help='path to data folder')

MAX_LEN=50
PERSONAGE_CONTENT = {'20-25',
 'Chinese',
 'English',
 'French',
 'Indian',
 'Italian',
 'Japanese',
 'lot',
 'small', 'amount',
 'average',
 'cheap',
 'city', 'centre',
 'coffee', 'shop',
 'decent',
 'excellent',
 'family',
 'fast', 'food',
 'friendly',
 'high',
 'kid',
 'low',
 'mediocre',
 'moderate',
 'nameVariable',
 'nearVariable',
 'pub',
 'restaurant',
 'riverside'}

def build_lm(data_folder):
    mdl = KneserNey(3, 4)
    for split in ['train', 'valid']:
        folder = os.path.join(data_folder, split)
        for file in os.scandir(folder):
            if 'x.csv' in file.name:
                with open(file.path, 'r') as f:
                    reader = csv.reader(f)
                    for line in reader:
                        text = " ".join(line) ##check if right
                        mdl.train(text.lower().strip().split())
    mdl.optimize()
    mdl.save(os.path.join(data_folder, 'knlm.model'))

def load_lm(data_folder):
    mdl = KneserNey.load(os.path.join(data_folder, 'knlm.model'))
    return mdl 

def get_perplexity(mdl, sent):
    if isinstance(sent, str):
        sent = sent.split()
    len_ = len(sent)
    ll = mdl.evaluate(sent)
    pp = pow(2, -1*(ll/len_))
    return pp


def get_vectors(inputs, seq_lens, f_labels, model, model_name):
    if model_name == 'baseline':
        _,z_c = model.E_c(inputs, seq_lens)
        z_f = model.E_f(f_labels)
    else:
        _,z_c = model.E_c(inputs, seq_lens)
        _,z_f = model.E_f(inputs, seq_lens)
    
    return z_c, z_f

def strip_eos(sents):
    return [sent[1:sent.index('<eos>')] if '<eos>' in sent else sent[1:]
        for sent in sents]

ends = ['<pad>', '<go>', '<eos>']
def strip_ends(sent):
    return [x for x in sent if x not in ends]

def get_sents(inputs, vocab):
    ip_arr = inputs.t().detach().cpu().numpy()
    sents = []
    for row in ip_arr:
        words = [vocab.idx2word[i] for i in row]
        sents.append(strip_ends(words))
    
    return sents

def get_outputs(outputs, vocab):
    op = outputs.detach().t().cpu().numpy()
    sents = [[vocab.idx2word[i] for i in sent] for sent in op]
    sents = strip_eos(sents)
    sents = [strip_ends(words) for words in sents]
    #sents = [" ".join(x) for x in ]
    return sents

def get_generated(z_c, z_f, model, alg='top5', max_len=MAX_LEN):
    
    sents = model.Gen.generate(z_c, z_f, max_len, alg=alg)

    return sents

def get_self_bleu(hyp, ref, gram=3):
    w = 1.0/gram
    weights = [w for _ in range(gram)]
    return bleu_score.sentence_bleu([ref], hyp, weights=weights)

def strip_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation))

#fasttext
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

#Rerieval 
def get_neighbors(sim_matrix, row_ind, k=5):

    row = sim_matrix[row_ind]
    row[row_ind] = 0.
    args = np.argsort(row)[::-1][:k]

    return args

def mask_content(sent, terms):
    return [x for x in sent if strip_punct(x) not in terms ]

def get_content(sent, terms):
    return [x for x in sent if strip_punct(x) in terms]

def get_overlap_score(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    try:
        return 2*(len(set1.intersection(set2)))/(len(set1)+len(set2))
    except:
        return 0

def form_sim(s1, s2):
    fs1 = mask_content(s1, PERSONAGE_CONTENT)
    fs2 = mask_content(s2, PERSONAGE_CONTENT)
    
    return get_overlap_score(fs1, fs2)

def cont_sim(s1, s2):
    fs1 = get_content(s1, PERSONAGE_CONTENT)
    fs2 = get_content(s2, PERSONAGE_CONTENT)
    
    return get_overlap_score(fs1, fs2)

def form_similarity(fl1, fl2):
    if fl1 == fl2:
        return 1
    else:
        return 0

def get_content_vector_sim(v1, v2):
    match = []
    for a1, a2 in zip(v1, v2):
        if a1 == a2:
            match.append(1)
        else:
            match.append(0)
    
    return np.sum(match)/len(match)

def check_paraphrase(sim_matrix, row_ind, k=1):
    row = sim_matrix[row_ind]
    
    topk = np.argsort(row)[::-1][:k]
    
    if row_ind in topk:
        return 1
    else:
        return 0


def transfer_style(sents1, sents2, seq_lens1, seq_lens2, flabels1, flabels2, model, model_name):
    zc1, zf1 = get_vectors(sents1, seq_lens1, flabels1, model, model_name)
    zc2, zf2 = get_vectors(sents2, seq_lens2, flabels2, model, model_name)

    op1 = get_generated(zc1, zf2, model, alg='greedy', max_len=MAX_LEN)
    op2 = get_generated(zc2, zf1, model, alg='greedy', max_len=MAX_LEN)

    return op1, op2

def get_content_words(sents, cont_dict, field='key'):

    if field == 'key':
        keys = set(list(cont_dict.keys()))
    else:
        values = []
        for key, vals in cont_dict.items():
            values.extend(vals)
        keys = set(values)

    words = []
    for op in sents:
        assert isinstance(op, list), "wrong format"
        terms = set(op)
        inter = terms.intersection(keys)
        words.append(inter)
    
    return words

def get_overlap_score(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    return 2*(len(set1.intersection(set2)))/(len(set1)+len(set2))

def get_content_vector(overlaps, map_):
    vecs = []
    for row in overlaps:
        vc = [map_[w] for w in row]
        vecs.append(vc)
    return np.array(vecs)



def get_test_batches(args, vocab):

    test_sents1, test_sents2, test_flabels1, test_flabels2 = load_parallel('data/personage-nlg/parallel/test/')
    test_clabels = load_labels('data/personage-nlg/parallel/test/clabels.csv')
    test_clabels1 = test_clabels
    test_clabels2 = test_clabels

    batcher = MultLabelled()
    test_batches1, _ = batcher.get_batches(test_sents1, test_flabels1, test_clabels1, vocab, args.batch_size, args.device)

    test_batches2, _ = batcher.get_batches(test_sents2, test_flabels2, test_clabels2, vocab, args.batch_size, args.device)

    return test_batches1, test_batches2

def get_dae_outputs(args):
    print(args.save_dir, args.model_type)
    log_file = os.path.join(args.save_dir, "eval_log.txt")
    data_folder = args.data_folder
    #data_folder = '/h/vkpriya/data/personage-nlg/labelled'
    test_path = os.path.join(data_folder, 'test')
    test_sents, test_flabels, test_clabels = load_data(test_path)
    
    logging('# test sents {}, tokens {}'.format(
        len(test_sents), sum(len(s) for s in test_sents)), log_file)


    vocab_file = os.path.join(args.save_dir, 'vocab.txt')

    # if not os.path.isfile(vocab_file):
    #     Vocab.build(train_sents, vocab_file, args.vocab.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)

    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    args.device = device

    batcher = MultLabelled()
    test_batches, _ = batcher.get_batches(test_sents, test_flabels, test_clabels, vocab, args.batch_size, args.device)
    logging("Number of test batches: {}".format(len(test_batches)), log_file)
    model_name = ''
    model = None 

    if args.model_type == 'para':
        model = pLinAE(vocab, args.model)
        model_name = 'para'
    
    elif args.model_type == 'baseline':
        model = CondDecode(vocab, args.model)
        model_name = 'baseline'
    
    elif args.model_type == 'disae':
        model = linAE(vocab, args.model)
        model_name = 'disae'

    print(model_name)

    name = 'ckpt.pt'
    path = os.path.join(args.save_dir, name)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    model.to(args.device)
    model.eval()

    form_labels = []
    cont_labels = []
    in_sents = []
    out_sents = []
    f_vecs = []
    c_vecs = []
    for inputs, targets, seq_lens, f_labels, c_labels in test_batches:
        in_sents.extend(get_sents(inputs, vocab))
        
        z_c, z_f = get_vectors(inputs, seq_lens, f_labels, model, model_name)
        f_vecs.extend(z_f.detach().cpu().numpy())
        c_vecs.extend(z_c.detach().cpu().numpy())
        form_labels.extend(f_labels.cpu().tolist())
        cont_labels.extend(c_labels.cpu().t().numpy())
        
        outputs = get_generated(z_c, z_f, model)
        
        out_sents.extend(get_outputs(outputs, vocab))

    res = {
        'form_labels': form_labels,
        'cont_labels': cont_labels,
        'in_sents': in_sents,
        'out_sents': out_sents,
        'f_vecs': f_vecs,
        'c_vecs': c_vecs
    }

    return res

def ae_outputs(arg_str, model_type='disae'):
    parse_args = parser.parse_args(arg_str.split())
    config_file = parse_args.config
    config_name = parse_args.config_name
    chkpt_dir = parse_args.checkpoint_dir
    data_folder = parse_args.data_folder


    with open(config_file, 'r') as f:
        all_args = yaml.safe_load(f)

    args = get_args(all_args, config_name)
    args = Box(args)

    if chkpt_dir is not None:
        args.save_dir = chkpt_dir
    args.data_folder = data_folder
    args.model_type = model_type
    # if type == 'disae':
    res = get_dae_outputs(args)

    # else:
    #     res = get_pae_outputs(args)

    print(args.save_dir)

    return res

