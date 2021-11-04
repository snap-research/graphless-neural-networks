import copy
import torch
import dgl
import torch.nn.functional as F
from utils import set_seed

'''
1. Train and eval
'''


# +
def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    '''
    GNN full-batch training. Input the entire graph `g` as data. 
    lamb: weight parameter lambda
    '''
    model.train()

    # Compute loss and prediction
    logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = lamb * criterion(out[idx_train], labels[idx_train])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    '''
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`. 
    lamb: weight parameter lambda
    '''
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        
        loss = lamb * criterion(out, batch_labels)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# -

def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    '''
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    '''
    model.eval()
    with torch.no_grad():
        logits = model.inference(data, feats)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss.item(), score

'''
2. Run teacher
'''


def run_transductive(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger, loss_and_score):
    '''
    Train and eval under the transductive setting. 
    The train/valid/test split is specified by `indices`.
    loss_and_score: Stores losses and scores.
    '''
    set_seed(conf['seed'])
    device = conf['device']
    idx_train, idx_val, idx_test = indices
    
    feats = feats.to(device)
    labels = labels.to(device)

    if 'SAGE' in model.model_name:
        # Create dataloader for SAGE
        
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in conf['fan_out'].split(',')])
        dataloader = dgl.dataloading.NodeDataLoader(g, idx_train, sampler, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])
    
        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=False, drop_last=False, num_workers=conf['num_workers'])
        
        data = dataloader
        data_eval = dataloader_eval
    else:
        g = g.to(device)
        data = g
        data_eval = g
        
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        if 'SAGE' in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer)
        else: 
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)
            
        if epoch % conf['eval_interval'] == 0:
            out, loss_val, score_val = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_val)
            # Use evaluator instead of evaluate to avoid redundant forward pass
            score_train = evaluator(out[idx_train], labels[idx_train])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_train: {score_train:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            loss_and_score += [[epoch, loss, loss_val, score_train, score_val, score_test]]
                
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state) 
    out, loss_val, score_val = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_val)
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test

def run_inductive(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger, loss_and_score):
    '''
    Train and eval under the inductive setting. 
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.    
    '''
    set_seed(conf['seed'])
    device = conf['device']
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)
    
    if 'SAGE' in model.model_name:
        # Create dataloader for SAGE
        
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in conf['fan_out'].split(',')])
        obs_dataloader = dgl.dataloading.NodeDataLoader(obs_g, obs_idx_train, sampler, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(obs_g, torch.arange(obs_g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=False, drop_last=False, num_workers=conf['num_workers'])
        dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=False, drop_last=False, num_workers=conf['num_workers'])
        
        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
        
    else:
        obs_g = obs_g.to(device)
        g = g.to(device)
        
        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g
        
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        if 'SAGE' in model.model_name:
            loss = train_sage(model, obs_data, obs_feats, obs_labels, criterion, optimizer)
        else:
            loss = train(model, obs_data, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            
        if epoch % conf['eval_interval'] == 0:
            obs_out, loss_val, score_val = evaluate(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
            # Use evaluator instead of evaluate to avoid redundant forward pass
            score_train = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            
            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_test_ind)

            logger.debug(f'Ep {epoch: 3d} | l: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}')
            loss_and_score += [[epoch, loss, loss_val, score_train, score_val, score_test_tran, score_test_ind]]

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    obs_out, loss_val, score_val = evaluate(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out, loss_test_ind, score_test_ind = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_test_ind)
    out[idx_obs] = obs_out
    return out, score_val, score_test_tran, score_test_ind




# +
'''
3. Distill
'''
def distill_run_transductive(conf, model, g, feats, labels, out_t, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger, loss_and_score):
    '''
    Distill training and eval under the transductive setting. 
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.    
    '''

    set_seed(conf['seed'])
    device = conf['device']
    lamb = conf['lamb']
    idx_l, idx_t, idx_val, idx_test = distill_indices
 
    g = g.to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    out_t = out_t.to(device)
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss_l = train(model, g, feats, labels, criterion_l, optimizer, idx_l, lamb)
        loss_t = train(model, g, feats, out_t, criterion_t, optimizer, idx_t, 1 - lamb)
        loss = loss_l + loss_t
        if epoch % conf['eval_interval'] == 0:
            out, loss_val, score_val = evaluate(model, g, feats, labels, criterion_l, evaluator, idx_val)
            # Use evaluator instead of evaluate to avoid redundant forward pass
            score_l = evaluator(out[idx_l], labels[idx_l])
            score_t = evaluator(out[idx_t], labels[idx_t])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_l: {score_l:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            loss_and_score += [[epoch, loss, loss_val, score_l, score_val, score_test]]

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    out, loss_val, score_val = evaluate(model, g, feats, labels, criterion_l, evaluator, idx_val)
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test

def distill_run_inductive(conf, model, g, feats, labels, out_t, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger, loss_and_score):
    '''
    Distill training and eval under the inductive setting. 
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.    
    '''

    set_seed(conf['seed'])
    device = conf['device']
    lamb = conf['lamb']
    obs_idx_l, obs_idx_t, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = distill_indices

    obs_g = g.subgraph(idx_obs).to(device)
    g = g.to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t[idx_obs].to(device)

    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss_l = train(model, obs_g, obs_feats, obs_labels, criterion_l, optimizer, obs_idx_l, lamb)
        loss_t = train(model, obs_g, obs_feats, obs_out_t, criterion_t, optimizer, obs_idx_t, 1 - lamb)
        loss = loss_l + loss_t
        
        if epoch % conf['eval_interval'] == 0:
            obs_out, loss_val, score_val = evaluate(model, obs_g, obs_feats, obs_labels, criterion_l, evaluator, obs_idx_val)
            # Use evaluator instead of evaluate to avoid redundant forward pass
            score_l = evaluator(obs_out[obs_idx_l], obs_labels[obs_idx_l])
            score_t = evaluator(obs_out[obs_idx_t], obs_labels[obs_idx_t])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(model, g, feats, labels, criterion_l, evaluator, idx_test_ind)

            logger.debug(f'Ep {epoch: 3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}')
            loss_and_score += [[epoch, loss, loss_val, score_l, score_val, score_test_tran, score_test_ind]]

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    
    obs_out, loss_val, score_val = evaluate(model, obs_g, obs_feats, obs_labels, criterion_l, evaluator, obs_idx_val)
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out, loss_test_ind, score_test_ind = evaluate(model, g, feats, labels, criterion_l, evaluator, idx_test_ind)
    out[idx_obs] = obs_out
    return out, score_val, score_test_tran, score_test_ind
