# +
import argparse
import numpy as np
import dgl
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t
from utils import get_logger, get_evaluator, set_seed, get_training_config, check_writable, check_readable, compute_min_cut_loss
from train_and_eval import distill_run_transductive


# -

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DGL implementation')
    parser.add_argument('--device', type=int, default=2, help='CUDA device')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_level', type=int, default=20, help='Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}')
    parser.add_argument('--console_log', action='store_true', 
                        help='Set to True to display log info in console')
    parser.add_argument('--output_path', type=str, default='outputs', help='Path to save outputs')
    parser.add_argument('--num_exp', type=int, default=1, help='Repeat how many experiments')

    '''Dataset'''
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to data')
    parser.add_argument('--labelrate_train', type=int, default=20, help='How many labeled data per class as train set')
    parser.add_argument('--labelrate_val', type=int, default=30, help='How many labeled data per class in valid set')

    '''Model and Optimization'''
    parser.add_argument('--model_config_path', type=str, default='./train.conf.yaml', help='Path to model configeration')
    parser.add_argument('--teacher', type=str, default='SAGE', help='Teacher model')
    parser.add_argument('--student', type=str, default='MLP', help='Student model')
    parser.add_argument('--num_layers', type=int, default=2, help='Student model number of layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Student model hidden layer dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dropout_ratio', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--norm_type', type=str, default='none', help='One of [none, batch, layer]')
    parser.add_argument('--max_epoch', type=int, default=500, help='Evaluate once per how many epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stop is the score on validation set does not improve for how many epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate once per how many epochs')
    
    '''Others'''
    parser.add_argument('--lamb', type=float, default=0, 
                        help='Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]')
    parser.add_argument('--feature_noise', type=float, default=0, 
                        help='add white noise to features for analysis, value in [0, 1] for noise level')
        
    args = parser.parse_args()
    
    assert(1 <= args.num_exp)
    assert(0 <= args.feature_noise <= 1)
    
    if args.feature_noise != 0:
        args.output_path += f'_noisy_features/noise_{args.feature_noise}/'
         
    return args


def run(args):
    ''' Set seed, device, and logger '''
    set_seed(args.seed)
    device = torch.device('cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu')
    output_dir = Path.cwd().joinpath(args.output_path, 'transductive', args.dataset, f'{args.teacher}_{args.student}', f'seed_{args.seed}')
    check_writable(output_dir, overwrite=False)
    out_t_dir = Path.cwd().joinpath(args.output_path, 'transductive', args.dataset, args.teacher, f'seed_{args.seed}')
    check_readable(out_t_dir)
    logger = get_logger(output_dir.joinpath('log'), args.console_log, args.log_level)
    logger.info(f'output_dir: {output_dir}')
    logger.info(f'out_t_dir: {out_t_dir}')

    ''' Load data and model config'''
    g, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.seed, args.labelrate_train, args.labelrate_val, args.data_path)
    logger.info(f'Total {g.number_of_nodes()} nodes.')
    logger.info(f'Total {g.number_of_edges()} edges.')

    feats = g.ndata['feat']
    args.feat_dim = g.ndata['feat'].shape[1]
    args.label_dim = labels.max().item() + 1
    
    if 0 < args.feature_noise <= 1:
        feats = (1 - args.feature_noise)* feats + args.feature_noise * torch.randn_like(feats)        

    ''' Model config '''
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.student, args.dataset) # Note: student config
    conf = dict(args.__dict__, **conf)
    conf['device'] = device
    logger.info(f'conf: {conf}')

    ''' Model init '''
    model = Model(conf)
    optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'], weight_decay=conf['weight_decay'])
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    evaluator = get_evaluator()
    idx_l = idx_train
    idx_t = torch.cat([idx_train, idx_test])
    distill_indices = (idx_l, idx_t, idx_val, idx_test)

    '''Load teacher model output'''
    out_t = load_out_t(out_t_dir)
    logger.debug(f'teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}')
    logger.debug(f'teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}')
    logger.debug(f'teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}')

    ''' Run '''
    loss_and_score = []
    out, score_val, score_test = distill_run_transductive(conf, model, g, feats, labels, out_t, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger, loss_and_score)

    logger.info(f"Teacher: {conf['teacher']}. Student: {conf['student']}. Dataset: {conf['dataset']}. Lamb: {conf['lamb']}")
    logger.info(f"Best valid model on test set: score_val: {score_val :.4f}, score_test: {score_test :.4f}")
    logger.info(f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}" )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    ''' Saving results '''
    # Student output
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath('out'), out_np)
    
    # Loss curves
    loss_and_score = np.array(loss_and_score)
    np.savez(output_dir.joinpath('loss_and_score'), loss_and_score)

    # Model
    torch.save(model.state_dict(), output_dir.joinpath('model'))

    # Test result
    with open(output_dir.parent.joinpath('test_results'), 'a+') as f:
        f.write(f"{score_test :.4f}\n")

    # Min-cut loss
    min_cut = compute_min_cut_loss(g, out)
    with open(output_dir.parent.joinpath('min_cut_loss'), 'a+') as f:
        f.write(f"{min_cut :.4f}\n")

    return score_test


# +
def repeat_run(args):
    s = []
    for seed in range(args.num_exp):
        args.seed = seed
        score_t = run(args)
        s += [score_t]
    score_test = np.array(s)
    return score_test.mean(), score_test.std()
    
def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        print(f'score: {score: .4f}')
    elif args.num_exp > 1:
        score_test_mean, score_test_std = repeat_run(args)
        print(f'{score_test_mean : .4f}  {score_test_std : .4f}')
        


# -

if __name__ == "__main__":
    main()












