#!/bin/bash

# Train GLNN with SAGE teacher on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"

aggregated_result_file="glnn_cpf.txt"
for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        python train_student.py --exp_setting $e --teacher "SAGE" --dataset $ds --num_exp 10 \
                                --max_epoch 200 --patience 50 \
                                --save_results >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file    
done
