#!/bin/bash
mkdir -p ./logs/LongForecasting/exchange_rate

export CUDA_VISIBLE_DEVICES=0
model_name=LiNo
seq_lens=(96)
bss=(32)
lrs=(1e-4)
log_dir="./logs/LongForecasting/exchange_rate/"
layers=(2)
pred_lens=(96)
dropouts=( 0.)
d_models=(256)

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
                            --root_path ./dataset/exchange_rate/ \
                            --data_path exchange_rate.csv \
                            --model_id "exchange_rate_${seq_len}_${pred_len}" \
                            --model $model_name \
                            --data custom \
                            --features M \
                            --seq_len $seq_len \
                            --pred_len $pred_len \
                            --batch_size $bs \
                            --learning_rate $lr \
                            --layers $layer\
                            --dropout $dropout\
                            --d_model $d_model\
                            --enc_in 8 \
                            --dec_in 8 \
                            --c_out 8 \
                            --lradj type2\
                            --patience 6\
                            --des 'Exp' \
                            --itr 1 >"${log_dir}bs${bs}_lr${lr}_lay${layer}_dp${dropout}_dm${d_model}_${pred_len}_${seq_len}.log"
                        done
                    done
                done
            done
        done
    done
done
