import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, norm_type='none'):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.norm_type = norm_type

        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == 'layer':
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h_list = [feats]
        for i, layer in enumerate(self.layers[:-1]):
            h = h_list[-1]
            h = layer(h) 
            h_list.append(h)
            
            if self.norm_type != 'none':
                h = self.norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.layers[-1](h)
        return h_list, h


'''
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
'''
class SAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, norm_type='none'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, 'gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, 'gcn'))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, 'gcn'))
            self.layers.append(SAGEConv(hidden_dim, output_dim, 'gcn'))
            
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = activation

    def forward(self, blocks, feats):
        h_list = [feats]
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = h_list[-1]

            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            h_list.append(h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h

    def inference(self, dataloader, feats):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        dataloader : the entire graph loaded in blocks with full neighbors for each node.
        feats : the input feats of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        device = feats.device
            
        for l, layer in enumerate(self.layers):
            y = torch.zeros(feats.shape[0], self.hidden_dim if l != len(self.layers) - 1 else self.output_dim).to(device)
            for input_nodes, output_nodes, blocks in dataloader:

                block = blocks[0].int().to(device)

                h = feats[input_nodes]
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            feats = y
        return y


class Model(nn.Module):
    '''
    Wrapper of different GNN models
    '''
    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf['model_name']
        if 'MLP' in conf['model_name']:
            self.encoder = MLP(num_layers=conf['num_layers'],
                        input_dim=conf['feat_dim'],
                        hidden_dim=conf['hidden_dim'],
                        output_dim=conf['label_dim'],
                        dropout_ratio=conf['dropout_ratio'],
                        norm_type=conf['norm_type']).to(conf['device'])
        elif 'SAGE' in conf['model_name']:
            self.encoder = SAGE(num_layers=conf['num_layers'],
                input_dim=conf['feat_dim'],
                hidden_dim=conf['hidden_dim'],
                output_dim=conf['label_dim'],
                dropout_ratio=conf['dropout_ratio'],
                activation=F.relu,
                norm_type=conf['norm_type']).to(conf['device'])
   
    def forward(self, data, feats):
        '''
        data: a graph `g` or a `dataloader` of blocks
        '''
        if 'MLP' in self.model_name:
            return self.encoder(feats)[1]
        else:
            return self.encoder(data, feats)[1]
        
    def forward_fitnet(self, data, feats):
        if 'MLP' in self.model_name:
            return self.encoder.forward(feats)
        else:
            return self.encoder.forward(data, feats)
    
    def inference(self, data, feats):
        if 'SAGE' in self.model_name:
            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)



    
