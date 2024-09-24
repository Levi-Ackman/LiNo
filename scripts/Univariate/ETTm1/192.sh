#!/bin/bash
mkdir -p ./logs/LongForecasting/ETTm1

export CUDA_VISIBLE_DEVICES=5
model_name=LiNo
seq_lens=(96)
bss=(128)
lrs=(1e-4)
log_dir="./logs/LongForecasting/ETTm1/"
layers=(2)
pred_lens=(192)
dropouts=(0.)
d_models=(512)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for layer in "${layers[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for d_model in "${d_models[@]}"; do
                    for pred_len in "${pred_lens[@]}"; do
                        for seq_len in "${seq_lens[@]}"; do
                            python -u run.py \
                            --task_name long_term_forecast \
                            --is_training 1 \
                            --root_path /data/gqyu/dataset/ETT-small/ \
                            --data_path ETTm1.csv \
                            --model_id "ETTm1_${seq_len}_${pred_len}" \
                            --model $model_name \
                            --data ETTm1 \
                            --features S \
                            --seq_len $seq_len \
                            --pred_len $pred_len \
                            --batch_size $bs \
                            --learning_rate $lr \
                            --layers $layer\
                            --dropout $dropout\
                            --d_model $d_model\
                            --enc_in 1 \
                            --dec_in 1 \
                            --c_out 1 \
                            --des 'Exp' \
                            --patience 6\
                            --itr 1 >"${log_dir}bs${bs}_lr${lr}_lay${layer}_dp${dropout}_dm${d_model}_${pred_len}_${seq_len}.log"
                        done
                    done
                done
            done
        done
    done
done
