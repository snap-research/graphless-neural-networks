import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset


def load_data(dataset, data_path='./data/'):
    data = DglNodePropPredDataset(dataset, data_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    
    g, labels = data[0]
    labels = labels.squeeze()
    
    if dataset == 'ogbn-arxiv':
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()

    return g, labels, idx_train, idx_val, idx_test    


def load_out_t(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir.joinpath('out.npz'))['arr_0'])
