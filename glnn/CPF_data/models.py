import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError
from dgl.ops import edge_softmax
from dgl.nn.pytorch.conv import APPNPConv
from dgl.nn.pytorch import GraphConv, SAGEConv


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
        dataloader : The entire graph loaded in blocks with full neighbors for each node.
        feats : The input feats of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and layers.
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


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, norm_type='none'):
        super().__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.convs = nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.norm_type = norm_type
        
        if num_layers == 1:
            self.convs.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.convs.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
                if self.norm_type == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == 'layer':
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.convs.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h_list = [feats]
        for i, conv in enumerate(self.convs[:-1]):
            h = h_list[-1]
            h = conv(g, h)
            h_list.append(h)
            
            if self.norm_type != 'none':
                h = self.norms[i](h)
            h = self.dropout(h)
        h = self.convs[-1](g, h)
        return h_list, h


# +
'''
Adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/models
'''
class GAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation,
                  num_heads=8, attn_drop=0.3, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        
        # Hard coded values from the CPF implementation
        num_layers = 1
        num_out_heads = 1
        hidden_dim //= num_heads
        
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        
        heads = ([num_heads] * num_layers) + [num_out_heads]
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            input_dim, hidden_dim, heads[0],
            dropout_ratio, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l-1], hidden_dim, heads[l],
                dropout_ratio, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            hidden_dim * heads[-2], output_dim, heads[-1],
            dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h_list = [feats]
        for l in range(self.num_layers):
            h = h_list[-1]
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h, att = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h_list.append(h)
            
        # output projection
        h, att = self.gat_layers[-1](g, h)
        h = h.mean(1)
        return h_list, h

class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dropout_ratio=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.dropout_ratio = nn.Dropout(dropout_ratio)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src = h_dst = self.dropout_ratio(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            att = graph.edata['a'].squeeze()
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, att


# +
class APPNP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, norm_type='none',
                edge_drop=0.5, alpha=0.1, k=10):
    
        super(APPNP, self).__init__()
        
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

        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h_list = [feats]
        for i, layer in enumerate(self.layers[:-1]):
            h = h_list[-1]
            h = layer(h)
            h_list.append(h)
            
            if self.norm_type != 'none':
                h = self.norms[i](h)
            h = self.dropout(h)
        h = self.layers[-1](h)
        h = self.propagate(g, h)
        return h_list, h

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
        elif 'GCN' in conf['model_name']:
            self.encoder = GCN(num_layers=conf['num_layers'],
                input_dim=conf['feat_dim'],
                hidden_dim=conf['hidden_dim'],
                output_dim=conf['label_dim'],
                dropout_ratio=conf['dropout_ratio'],
                activation=F.relu,
                norm_type=conf['norm_type']).to(conf['device'])
        elif 'GAT' in conf['model_name']:
            self.encoder = GAT(num_layers=conf['num_layers'],
                        input_dim=conf['feat_dim'],
                        hidden_dim=conf['hidden_dim'],
                        output_dim=conf['label_dim'],
                        dropout_ratio=conf['dropout_ratio'],
                        activation=F.relu,
                        attn_drop=conf['attn_dropout_ratio']).to(conf['device'])
        elif 'APPNP' in conf['model_name']:
            self.encoder = APPNP(num_layers=conf['num_layers'],
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
# -




