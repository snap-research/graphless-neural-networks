# -*- coding: utf-8 -*-
# +
import time
import numpy as np
import torch
import logging
import pytz
import itertools
import random
import os
import copy
import yaml
import dgl
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator

'''
1. general

'''
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def idx_split(idx, ratio, seed=0):
    '''
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    '''
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n) 
    
    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2

def get_training_config(config_path, model_name, dataset):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = dict(full_config['global'], **full_config[dataset]['common'])
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config
        
    specific_config['model_name'] = model_name
    return specific_config
    

def check_writable(dir, overwrite=True):
    import shutil
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite:
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        pass

def check_readable(dir):
    if not os.path.exists(dir):
        print(dir)
        raise ValueError(f'No such a directory or file!')
    
def timetz(*args):
    tz = pytz.timezone('US/Pacific')
    return datetime.now(tz).timetuple()

def get_logger(filename, console_log=False, log_level=logging.INFO):
    tz = pytz.timezone('US/Pacific')
    log_time = datetime.now(tz).strftime('%b%d_%H_%M_%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    
    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%b%d %H-%M-%S')
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger



# -
def graph_split(idx_train, idx_val, idx_test, rate, seed):
    '''
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)

        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    '''
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1:N1+N2]
    obs_idx_test = obs_idx_all[N1+N2:]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind



# +
'''
2. Metrics
'''
def get_evaluator(name='ogbn-arxiv'):
    ogb_evaluator = Evaluator(name)
    if name in ['ogbn-arxiv', 'ogbn-products']:
        def evaluator(out, labels):
            pred = out.argmax(1, keepdim=True)
            input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
            return ogb_evaluator.eval(input_dict)['acc']
        
#         def evaluator(out, labels):
#             pred = out.argmax(1)
#             return pred.eq(labels).float().mean().item()
    else:
        raise ValueError("Unknown dataset")
        
    return evaluator
# -


