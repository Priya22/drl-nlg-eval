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
from eval_utils import *
from sklearn.metrics.pairwise import cosine_similarity

root_dir = 'results/nlg/' #saved models
data_folder = 'data/personage-nlg/labelled/'

config_names = ['ae', 'all', 'content', 'form', 'mot', 'para', 'para_f', 'baseline']
config_types = ['disae', 'disae', 'disae', 'disae', 'disae', 'para', 'para', 'baseline']

def get_results(args=None):

    res = {}

    for config, ctype in zip(config_names, config_types):
        chkpt_dir = os.path.join(root_dir, config)
        print(config, ctype)

        arg_str = " --config /h/vkpriya/disentanglement_losses/config/nlg_lin/config.yaml --config_name " + config + " --checkpoint_dir " + chkpt_dir + ' --data_folder ' +  data_folder
        
        if config=='baseline':
            arg_str = " --config /h/vkpriya/disentanglement_losses/config/nlg_lin/baseline.yaml --config_name all" + " --checkpoint_dir " + chkpt_dir + ' --data_folder ' +  data_folder
            
        res[config] = ae_outputs(arg_str, ctype)
    

    return res

def eval_retrieval(res):
    ret_scores = {}

    for config in config_names:
        ret_scores[config] = {}
        
        for v_name in ['f_vecs', 'c_vecs']:
            form_scores = []
            cont_scores = []
            
            base_form = []
            base_cont = []
            
            vecs = res[config][v_name]
            sim_matrix = cosine_similarity(vecs)

            for i in range(len(sim_matrix)):
                f_sc = []
                c_sc = []
                
                b_f = []
                b_c = []
                
                ip = res[config]['in_sents'][i]
                ip_cvec = res[config]['cont_labels'][i]
                
                nbs = get_neighbors(sim_matrix, i, k=5)
                for n in nbs:
                    op = res[config]['in_sents'][n]
                    op_cvec = res[config]['cont_labels'][n]
                    
                    f_sc.append(form_sim(ip, op))
                    c_sc.append(get_content_vector_sim(ip_cvec, op_cvec))
                    b_f.append(form_sim(ip, ip))
                    b_c.append(get_content_vector_sim(ip_cvec, ip_cvec))
                    
                form_scores.append(np.mean(f_sc))
                cont_scores.append(np.mean(c_sc))
                base_form.append(np.mean(b_f))
                base_cont.append(np.mean(b_c))
            
            ret_scores[config][v_name] = {}
            ret_scores[config][v_name]['f_scores'] = form_scores
            ret_scores[config][v_name]['c_scores'] = cont_scores
            ret_scores[config][v_name]['bf_scores'] = base_form
            ret_scores[config][v_name]['bc_scores'] = base_cont
    
    return ret_scores 