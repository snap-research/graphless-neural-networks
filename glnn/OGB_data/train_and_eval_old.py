import copy
import torch
import dgl
import torch.nn.functional as F
import numpy as np
from utils import set_seed


'''
1. tran, eval and distill
'''


def train(model, g, feats, labels, criterion, optimizer, idx_train, lamb=1):
    model.train()
    logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss = lamb * criterion(out[idx_train], labels[idx_train])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, feats, labels, evaluator, idx_eval=None):
    model.eval()
    with torch.no_grad():
        logits = model.inference(data, feats)
        out = logits.log_softmax(dim=1)

        if idx_eval is None:
            score = evaluator(out, labels)
        else:
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, score

def run_transductive(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device = conf['device']
    idx_train, idx_val, idx_test = indices
    
    g = g.to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
        if epoch % conf['eval_interval'] == 0:
            out, score_val = evaluate(model, g, feats, labels, evaluator, idx_val)
            score_train = evaluator(out[idx_train], labels[idx_train])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_train: {score_train:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    out, score_val = evaluate(model, g, feats, labels, evaluator, idx_val)
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test

def run_inductive(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device = conf['device']
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    
    obs_g = g.subgraph(idx_obs).to(device)
    g = g.to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
        if epoch % conf['eval_interval'] == 0:
            obs_out, score_val = evaluate(model, obs_g, obs_feats, obs_labels, evaluator, obs_idx_val)    
            score_train = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            
            # Evaluate the inductive part with the full graph
            out, score_test_ind = evaluate(model, g, feats, labels, evaluator, idx_test_ind)

            logger.debug(f'Ep {epoch: 3d} | l: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}')

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    obs_out, score_val = evaluate(model, obs_g, obs_feats, obs_labels, evaluator, obs_idx_val)
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out, score_test_ind = evaluate(model, g, feats, labels, evaluator, idx_test_ind)
    out[idx_obs] = obs_out
    return out, score_val, score_test_tran, score_test_ind



# +
def distill_run_transductive(conf, model, g, feats, labels, out_t, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger):
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
            out, score_val = evaluate(model, g, feats, labels, evaluator, idx_val)            
            score_l = evaluator(out[idx_l], labels[idx_l])
            score_t = evaluator(out[idx_t], labels[idx_t])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug('Ep %3d | l: %.3f | l_l: %.3f | l_t: %.3f | s_tr: %.2f | s_v: %.2f | s_tt: %.2f' % (
            epoch, loss, loss_l, loss_t, score_l*100, score_val*100, score_test*100))
                
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    out, score_val = evaluate(model, g, feats, labels, evaluator, idx_val)            
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test

def distill_run_inductive(conf, model, g, feats, labels, out_t, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device     = conf['device']
    lamb       = conf['lamb']
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
            obs_out, score_val = evaluate(model, obs_g, obs_feats, obs_labels, evaluator, obs_idx_val)
            score_l = evaluator(obs_out[obs_idx_l], obs_labels[obs_idx_l])
            score_t = evaluator(obs_out[obs_idx_t], obs_labels[obs_idx_t])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            # Evaluate the inductive part with the full graph
            _, score_test_ind = evaluate(model, g, feats, labels, evaluator, idx_test_ind)

            logger.debug('Ep %3d | l: %.3f | l_l: %.3f | l_t: %.3f | s_tr: %.2f | s_v: %.2f | s_tt: %.2f | s_ti: %.2f' % (
            epoch, loss, loss_l, loss_t, score_l*100, score_val*100, score_test_tran*100, score_test_ind*100))

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    obs_out, score_val = evaluate(model, obs_g, obs_feats, obs_labels, evaluator, obs_idx_val)
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    # Evaluate the inductive part with the full graph
    out, score_test_ind = evaluate(model, g, feats, labels, evaluator, idx_test_ind)
    out[idx_obs] = obs_out
    return out, score_val, score_test_tran, score_test_ind


# +
'''
2. tran, eval and distill using mini-batch
'''
def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.arange(feats.shape[0])

#     idx_batch = torch.randperm(feats.shape[0])[:num_batches * batch_size]
    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)
        
    total_loss = 0    
    for i in range(num_batches):
        logits = model(feats[idx_batch[i]])[1]
        out = F.log_softmax(logits, dim=1)
        loss = lamb * criterion(out, labels[idx_batch[i]])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        
    return total_loss / num_batches


# -

def eval_mini_batch(model, feats, labels, batch_size, evaluator):
    with torch.no_grad():
        model.eval()
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_all = []
        for i in range(num_batches):
            logits = model(feats[batch_size*i:batch_size*(i+1)])[1]
            out = logits.log_softmax(dim=1)
            out_all += [out.detach()]
        score = evaluator(torch.cat(out_all), labels)
    return score

def run_transductive_mini_batch(conf, model, feats, labels, indices, criterion, evaluator, optimizer, logger):
    '''
    For MLP only. Break nodes into minibatches ignoring their graph structure.
    model: an MLP
    '''
    set_seed(conf['seed'])
    device     = conf['device']
    batch_size = conf['batch_size']
    idx_train, idx_val, idx_test = indices
    
    feats = feats.to(device)
    labels = labels.to(device)

    feats_train, labels_train = feats[idx_train], labels[idx_train]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train_mini_batch(model, feats_train, labels_train, batch_size, criterion, optimizer)
        if epoch % conf['eval_interval'] == 0:
            score_train = eval_mini_batch(model, feats_train, labels_train, batch_size, evaluator)
            score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
            score_test = eval_mini_batch(model, feats_test, labels_test, batch_size, evaluator)

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_train: {score_train:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    score_train = eval_mini_batch(model, feats_train, labels_train, batch_size, evaluator)
    score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
    score_test = eval_mini_batch(model, feats_test, labels_test, batch_size, evaluator)
    return score_val, score_test

def run_inductive_mini_batch(conf, model, feats, labels, indices, criterion, evaluator, optimizer, logger):
    '''
    For MLP only. Break nodes into minibatches ignoring their graph structure.
    model: an MLP
    '''
    set_seed(conf['seed'])
    device     = conf['device']
    batch_size = conf['batch_size']
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    
    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]

    feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = obs_feats[obs_idx_test], obs_labels[obs_idx_test]
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train_mini_batch(model, feats_train, labels_train, batch_size, criterion, optimizer)
        if epoch % conf['eval_interval'] == 0:
            score_train = eval_mini_batch(model, feats_train, labels_train, batch_size, evaluator)
            score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
            score_test_tran = eval_mini_batch(model, feats_test_tran, labels_test_tran, batch_size, evaluator)
            score_test_ind = eval_mini_batch(model, feats_test_ind, labels_test_ind, batch_size, evaluator)

            logger.debug(f'Ep {epoch: 3d} | l: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}')
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
    score_test_tran = eval_mini_batch(model, feats_test_tran, labels_test_tran, batch_size, evaluator)
    score_test_ind = eval_mini_batch(model, feats_test_ind, labels_test_ind, batch_size, evaluator)
    # Since only used for MLP, no need to return the output
    return score_val, score_test_tran, score_test_ind


def distill_run_transductive_mini_batch(conf, model, feats, labels, out_t_all, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger):
    '''
    For MLP only. Break nodes into minibatches ignoring their graph structure.
    model: an MLP
    '''
    set_seed(conf['seed'])
    device     = conf['device']
    batch_size = conf['batch_size']
    lamb       = conf['lamb']
    idx_l, idx_t, idx_val, idx_test = (idx.to(device) for idx in distill_indices)
    
    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
        
    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss_l = train_mini_batch(model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb)
        loss_t = train_mini_batch(model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb)
        loss = loss_l + loss_t
        if epoch % conf['eval_interval'] == 0:
            score_l = eval_mini_batch(model, feats_l, labels_l, batch_size, evaluator)
            score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
            score_test = eval_mini_batch(model, feats_test, labels_test, batch_size, evaluator)

            logger.debug('Ep %3d | l: %.3f | l_l: %.3f | l_t: %.3f | s_tr: %.2f | s_v: %.2f | s_tt: %.2f' % (
            epoch, loss, loss_l, loss_t, score_l*100, score_val*100, score_test*100))
              
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
    score_test = eval_mini_batch(model, feats_test, labels_test, batch_size, evaluator)
    return score_val, score_test

def distill_run_inductive_mini_batch(conf, model, feats, labels, out_t_all, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger):
    '''
    For MLP only. Break nodes into minibatches ignoring their graph structure.
    model: an MLP
    '''
    set_seed(conf['seed'])
    device     = conf['device']
    batch_size = conf['batch_size']
    lamb       = conf['lamb']
    obs_idx_l, obs_idx_t, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]
    
    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = obs_feats[obs_idx_test], obs_labels[obs_idx_test]
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss_l = train_mini_batch(model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb)
        loss_t = train_mini_batch(model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb)
        loss = loss_l + loss_t
        if epoch % conf['eval_interval'] == 0:
            score_l = eval_mini_batch(model, feats_l, labels_l, batch_size, evaluator)
            score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
            score_test_tran = eval_mini_batch(model, feats_test_tran, labels_test_tran, batch_size, evaluator)
            score_test_ind  = eval_mini_batch(model, feats_test_ind, labels_test_ind, batch_size, evaluator)

            logger.debug('Ep %3d | l: %.3f | l_l: %.3f | l_t: %.3f | s_tr: %.2f | s_v: %.2f | s_tt: %.2f | s_ti: %.2f' % (
            epoch, loss, loss_l, loss_t, score_l*100, score_val*100, score_test_tran*100, score_test_ind*100))

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    score_val = eval_mini_batch(model, feats_val, labels_val, batch_size, evaluator)
    score_test_tran = eval_mini_batch(model, feats_test_tran, labels_test_tran, batch_size, evaluator)
    score_test_ind  = eval_mini_batch(model, feats_test_ind, labels_test_ind, batch_size, evaluator)
    return score_val, score_test_tran, score_test_ind


'''
3. tran and eval for GraphSAGE
'''

def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    total_loss = 0
    device = feats.device
    model.train()
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        
        loss = lamb * criterion(out, batch_labels)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def run_transductive_sage(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device = conf['device']
    
    idx_train, idx_val, idx_test = indices
    feats = feats.to(device)
    labels = labels.to(device)

    '''Create dataloader for SAGE'''
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()
#     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in conf['fan_out'].split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(g, idx_train, sampler, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])

    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train_sage(model, dataloader, feats, labels, criterion, optimizer)
        if epoch % conf['eval_interval'] == 0:
            out, score_val = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_val)
            score_train = evaluator(out[idx_train], labels[idx_train])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_train: {score_train:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    out, score_val = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_val)
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test


def run_inductive_sage(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device = conf['device']
    
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    obs_g = g.subgraph(idx_obs)
    obs_feats = feats[idx_obs].to(device)
    obs_labels = labels[idx_obs].to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    
    '''Create dataloader for SAGE'''
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    obs_g.create_formats_()
    g.create_formats_()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in conf['fan_out'].split(',')])
    obs_dataloader = dgl.dataloading.NodeDataLoader(obs_g, obs_idx_train, sampler, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])

    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    obs_dataloader_eval = dgl.dataloading.NodeDataLoader(obs_g, torch.arange(obs_g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])
    dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])
    
    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train_sage(model, obs_dataloader, obs_feats, obs_labels, criterion, optimizer)
        if epoch % conf['eval_interval'] == 0:
            obs_out, score_val = evaluate(model, obs_dataloader_eval, obs_feats, obs_labels, evaluator, obs_idx_val)
            score_train = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            
            # Evaluate the inductive part with the full graph
            out, score_test_ind = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_test_ind)

            logger.debug(f'Ep {epoch: 3d} | l: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}')

            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    
    obs_out, score_val = evaluate(model, obs_dataloader_eval, obs_feats, obs_labels, evaluator, obs_idx_val)
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out, score_test_ind = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_test_ind)
    out[idx_obs] = obs_out
    
    return out, score_val, score_test_tran, score_test_ind


'''
3.4 train, eval and distill for GraphSAGE with LADIES SAMPLING
'''
def compute_prob(g, seed_nodes, weight):
    out_frontier = dgl.reverse(dgl.in_subgraph(g, seed_nodes), copy_edata=True)
    if out_frontier.number_of_edges() == 0:
        return torch.zeros(g.number_of_nodes(), device=g.device), torch.zeros(0, device=g.device)

    if weight is None:
        edge_weight = torch.ones(out_frontier.number_of_edges(), device=out_frontier.device)
    else:
        edge_weight = out_frontier.edata[weight]
    with out_frontier.local_scope():
        # Sample neighbors on the previous layer
        out_frontier.edata['w'] = edge_weight
        out_frontier.edata['w'] = out_frontier.edata['w'] ** 2
        out_frontier.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'prob'))
        prob = out_frontier.ndata['prob']
        return prob

def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']

class LADIESNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, nodes_per_layer, weight=None, out_weight=None, replace=False):
        super().__init__(len(nodes_per_layer), return_eids=True)
        self.nodes_per_layer = nodes_per_layer
        self.weight = weight
        self.replace = replace
        self.out_weight = out_weight

    def sample_frontier(self, block_id, g, seed_nodes):
        if isinstance(seed_nodes, dict):
            seed_nodes = next(iter(seed_nodes.values()))
        
        num_nodes = self.nodes_per_layer[block_id]
        prob = compute_prob(g, seed_nodes, self.weight)
        candidate_nodes = torch.nonzero(prob, as_tuple=True)[0]

        if not self.replace and len(candidate_nodes) < num_nodes:
            neighbor_nodes = candidate_nodes
        else:
            neighbor_nodes = torch.multinomial(
                prob, self.nodes_per_layer[block_id], replacement=self.replace)
            
        neighbor_nodes = torch.cat([seed_nodes, neighbor_nodes])
        neighbor_nodes = torch.unique(neighbor_nodes)

        neighbor_graph = dgl.in_subgraph(g, seed_nodes)
        neighbor_graph = dgl.out_subgraph(neighbor_graph, neighbor_nodes)

        # Compute output edge weight
        if self.out_weight is not None:
            with neighbor_graph.local_scope():
                if self.weight is not None:
                    neighbor_graph.edata['P'] = neighbor_graph.edata[self.weight]
                else:
                    neighbor_graph.edata['P'] = torch.ones(neighbor_graph.number_of_edges(), device=neighbor_graph.device)
                neighbor_graph.ndata['S'] = prob
                neighbor_graph.apply_edges(dgl.function.e_div_u('P', 'S', 'P_tilde'))
                # Row normalize
                neighbor_graph.update_all(
                    dgl.function.copy_e('P_tilde', 'P_tilde'),
                    dgl.function.sum('P_tilde', 'P_tilde_sum'))
                neighbor_graph.apply_edges(dgl.function.e_div_v('P_tilde', 'P_tilde_sum', 'P_tilde'))
                w = neighbor_graph.edata['P_tilde']
            neighbor_graph.edata[self.out_weight] = w

        return neighbor_graph

def run_transductive_sage_ladies(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger):
    set_seed(conf['seed'])
    device = conf['device']
    
    idx_train, idx_val, idx_test = indices
    feats = feats.to(device)
    labels = labels.to(device)

    '''Create dataloader for SAGE with LADIES Sampler'''
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()
    g.edata['weight'] = normalized_laplacian_edata(g)

    sampler = LADIESNeighborSampler([512] * conf['num_layers'], weight='weight', out_weight='w', replace=False)
#     sampler = LADIESNeighborSampler([5000] * conf['num_layers'], weight='weight', out_weight='w', replace=False)
    dataloader = dgl.dataloading.NodeDataLoader(g, idx_train, sampler, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])
#     sampler_eval = sampler
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=conf['batch_size'], shuffle=True, drop_last=False, num_workers=conf['num_workers'])

    best_score_val, count = 0, 0
    for epoch in range(1, conf['max_epoch']+1):
        loss = train_sage(model, dataloader, feats, labels, criterion, optimizer)
        if epoch % conf['eval_interval'] == 0:
            out, score_val = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_val)
            score_train = evaluator(out[idx_train], labels[idx_train])
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(f'Ep {epoch: 3d} | loss: {loss:.4f} | score_train: {score_train:.4f} | score_val: {score_val:.4f} | score_test: {score_test:.4f}')
            if score_val >= best_score_val:
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf['patience'] or epoch == conf['max_epoch']:
            break

    model.load_state_dict(state)
    out, score_val = evaluate(model, dataloader_eval, feats, labels, evaluator, idx_val)
    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_val, score_test







def run_inductive(conf, model, g, feats, labels, indices, criterion, evaluator, optimizer, logger, loss_and_score):
    '''
    Train and eval under the inductive setting. 
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.    
    '''
        
    set_seed(conf['seed'])
    device = conf['device']
    batch_size=conf['batch_size']
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
        obs_dataloader = dgl.dataloading.NodeDataLoader(obs_g, obs_idx_train, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=conf['num_workers'])

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(obs_g, torch.arange(obs_g.num_nodes()), sampler_eval, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=conf['num_workers'])
        dataloader_eval = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler_eval, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=conf['num_workers'])
        
        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif 'MLP' in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]
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
        elif 'MLP' in model.model_name:
            loss = train_mini_batch(model, feats_train, labels_train, batch_size, criterion, optimizer)
        else:
            loss = train(model, obs_data, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            
        if epoch % conf['eval_interval'] == 0:
            if 'MLP' in model.model_name:
                obs_out, _, _ = evaluate_mini_batch(model, obs_feats, obs_labels, criterion, batch_size, evaluator)
                _, loss_val, score_val = evaluate_mini_batch(model, feats_val, labels_val, criterion, batch_size, evaluator)

                # Evaluate only the inductive part
                _, _, score_test_ind = evaluate_mini_batch(model, feats_test_ind, labels_test_ind, criterion, batch_size, evaluator)                
            else:
                obs_out, loss_val, score_val = evaluate(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
                # Evaluate the inductive part with the full graph
                out, loss_test_ind, score_test_ind = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_test_ind)

            # Use evaluator instead of evaluate to avoid redundant forward pass
            score_train = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
                
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
    if 'MLP' in model.model_name:
        obs_out, _, _ = evaluate_mini_batch(model, obs_feats, obs_labels, criterion, batch_size, evaluator)
        out, _, _ = evaluate_mini_batch(model, feats, labels, criterion, batch_size, evaluator)
        # Use evaluator instead of evaluate to avoid redundant forward pass
        score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
        score_test_ind = evaluator(out[idx_test_ind], labels[idx_test_ind])
        
    else:
        obs_out, _, score_val = evaluate(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
        out, _, score_test_ind = evaluate(model, data_eval, feats, labels, criterion, evaluator, idx_test_ind)

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out    
    return out, score_val, score_test_tran, score_test_ind
