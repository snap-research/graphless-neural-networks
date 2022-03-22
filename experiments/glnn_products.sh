#!/bin/bash

# Train GLNN with SAGE teacher on "ogbn-products"

for e in "tran" "ind"
do
    python train_student.py --exp_setting $e --teacher "SAGE" --student "MLP3w8" --dataset "ogbn-products" \
                        --num_exp 10 --max_epoch 200 --patience 30
done
