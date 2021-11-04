#!/bin/bash -ex
ds = 'ogbn-arxiv'
device = 0
m = 'SAGE'

python train_teacher_transductive.py --num_exp 2 --dataset $ds --seed 0 --device $device --teacher $m --patience 10 --log_level 10 --console_log --eval_interval 5
python train_student_transductive.py --num_exp 2 --dataset $ds --seed 0 --device $device --teacher $m --patience 10 --log_level 10 --console_log --eval_interval 5
python train_teacher_inductive.py --num_exp 2 --dataset $ds --seed 0 --device $device --teacher $m --patience 10 --log_level 10 --console_log --eval_interval 5
python train_student_inductive.py --num_exp 2 --dataset $ds --seed 0 --device $device --teacher $m --patience 10 --log_level 10 --console_log --eval_interval 5
