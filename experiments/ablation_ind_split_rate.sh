#!/bin/bash

# Train SAGE with 9 different inductive split rates: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# Then train corresponding GLNN for each split

aggregated_result_file="ablation_ind_split_rate.txt"
printf "Teacher\n" >> $aggregated_result_file    

for r in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    printf "%3s\n" $r >> $aggregated_result_file
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        python train_teacher.py --exp_setting "ind" --teacher "SAGE" --dataset $ds --split_rate $r \
                                --num_exp 5 --max_epoch 200 --patience 50 >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done

printf "Student\n" >> $aggregated_result_file

for r in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    printf "%3s\n" $r >> $aggregated_result_file
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        python train_student.py --exp_setting "ind" --teacher "SAGE" --dataset $ds --split_rate $r \
                                --num_exp 5 --max_epoch 200 --patience 50 >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done

