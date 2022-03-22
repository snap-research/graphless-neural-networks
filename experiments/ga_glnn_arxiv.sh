#!/bin/bash

# Train 1-hop GA-MLP and 1-hop GA-GLNN with SAGE teacher on "ogbn-arxiv" under the inductive setting

python train_teacher.py --exp_setting "ind" --teacher "MLP3w4" --dataset "ogbn-arxiv" \
                        --num_exp 5 --max_epoch 200 --patience 50 \
                        --feature_aug_k 1

python train_student.py --exp_setting "ind" --teacher "SAGE" --student "MLP3w4" --dataset "ogbn-arxiv" \
                    --num_exp 5 --max_epoch 200 --patience 50 \
                    --feature_aug_k 1

