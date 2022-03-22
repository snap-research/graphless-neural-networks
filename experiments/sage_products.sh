#!/bin/bash

# Train SAGE teacher on "ogbn-products"

for e in "tran" "ind"
do
    python train_teacher.py --exp_setting $e --teacher "SAGE" --dataset "ogbn-products" \
                            --num_exp 10 --max_epoch 40 --patience 10 \
                            --save_results
done
