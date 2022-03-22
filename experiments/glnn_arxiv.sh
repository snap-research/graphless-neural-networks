#!/bin/bash

# Train GLNN with SAGE teacher on "ogbn-arxiv"

for e in "tran" "ind"
do
    python train_student.py --exp_setting $e --teacher "SAGE" --student "MLP3w4" --dataset "ogbn-arxiv" \
                        --num_exp 10 --max_epoch 200 --patience 50
done
