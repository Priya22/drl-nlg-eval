import random
import numpy as np
import torch
import csv, os
from sklearn.model_selection import train_test_split

def set_seed(seed):     # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def strip_eos(sents):
    return [sent[1:sent.index('<eos>')] if '<eos>' in sent else sent[1:]
        for sent in sents]

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

def load_text(path):
    sents = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            sents.append(str(line[0]).split())
    
    return sents

def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            assert type(line) == list
            if len(line) == 1:
                labels.append(int(line[0]))
            else:
                labels.append([int(i) for i in line])
    
    return np.array(labels)
    
    # lbs = sorted(list(set(labels)))
    # l2i = {x:i for i,x in enumerate(lbs)}
    # labels = [l2i[x] for x in labels]

def load_data(folder):
    x_path = os.path.join(folder, 'x.csv')
    yf_path = os.path.join(folder, 'yf.csv')
    yc_path = os.path.join(folder, 'yc.csv')

    sents = load_text(x_path)
    yf = load_labels(yf_path)
    yc = load_labels(yc_path)

    return sents, yf, yc

def load_parallel(folder):
    sents1 = []
    sents2 = []
    labels1 = []
    labels2 = []
    
    x_path = os.path.join(folder, 'x.csv')
    with open(os.path.join(x_path), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            assert len(line) == 2
            l1, l2 = line
            sents1.append(l1.strip().split())
            sents2.append(l2.strip().split())
            
            
    y_path = os.path.join(folder, 'y.csv')
    with open(os.path.join(y_path), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            assert len(line) == 2
            l1, l2 = line
            labels1.append(int(l1))
            labels2.append(int(l2))
            
    return sents1, sents2, np.array(labels1), np.array(labels2)

def get_division(sents, labels, label_divison):
    '''
    label_division: a dictionary specifying which labels to use for train and test respectively. 
    example: {'train': [0,1,2], 'test': [3]}
    '''

    train_sents, train_labels, test_sents, test_labels = [], [], [], []

    for s, l in zip(sents, labels):
        if l in label_divison['train']:
            train_sents.append(s)
            train_labels.append(l)
        else:
            test_sents.append(s)
            test_labels.append(l)
    
    #get validation set
    train_x, valid_x, train_y, valid_y = train_test_split(train_sents, train_labels, stratify=train_labels)

    return {
        'train': [train_x, np.array(train_y)],
        'valid': [valid_x, np.array(valid_y)],
        'test': [test_sents, np.array(test_labels)]
    }



def load_parallel_clf(folder, with_domain=False):
    sents1, sents2, labels1, labels2 = load_parallel(folder)

    sents = sents1 + sents2
    labels = np.concatenate((labels1, labels2))

    # if with_domain:
    #     sents, labels = get_division(sents, labels)
    return sents, labels 



# def load_paraphrases(folder_path, split='train'):

#     x_path = os.path.join(folder_path, split + '_x.csv')
#     y_path = os.path.join(folder_path, split + '_y.csv')

#     sents1 = []
#     sents2 = []
#     labels1 = []
#     labels2 = []

#     with open(x_path, 'r') as f:
#         reader = csv.reader(f)
#         for line in reader:
#             sents1.append(line[0].split())
#             sents2.append(line[1].split())
    
#     with open(y_path, 'r') as f:
#         reader = csv.reader(f)
#         for line in reader:
#             labels1.append(line[0])
#             labels2.append(line[1])
    
#     lbs = list(set(labels2 + labels1))
#     l2i = {x:i for i,x in enumerate(lbs)}
#     print(l2i)
#     labels1 = [l2i[x] for x in labels1]
#     labels2 = [l2i[x] for x in labels2]

#     return sents1, sents2, labels1, labels2
    

def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')

def write_z(z, path):
    with open(path, 'w') as f:
        for zi in z:
            for zij in zi:
                f.write('%f ' % zij)
            f.write('\n')

# def logging(s, path, print_=True):
#     if print_:
#         print(s)
#     if path:
#         cwd = os.getcwd()
#         #print(cwd)
#         if not os.path.exists(cwd):
#             os.makedirs(cwd)
#         path = os.path.join(cwd, 'log.txt')
#         with open(path, 'a+') as f:
#             f.write(s + '\n')

def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        #cwd = os.getcwd()
        #print(cwd)
        # if not os.path.exists(cwd):
        #     os.makedirs(cwd)
        #path = os.path.join(cwd, 'log.txt')
        with open(path, 'a+') as f:
            f.write(s + '\n')

def lerp(t, p, q):
    return (1-t) * p + t * q

# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
    o = np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q)))
    so = np.sin(o)
    return np.sin((1-t)*o) / so * p + np.sin(t*o) / so * q

def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0*i/(n-1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)
