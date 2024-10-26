if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=MTST

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=custom

random_seed=2021
for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 2 \
      --n_heads 8 \
      --d_model 32 \
      --d_ff 64 \
      --dropout 0.1\
      --fc_dropout 0\
      --head_dropout 0\
      --patch_len_ls '8, 16' \
      --stride_ls '4, 8' \
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 10\
      --patience 3 \
      --itr 1 --batch_size 32 --learning_rate 0.0001
#      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
#      --use_mlflow \
#      --rel_pe 'rel_sin' \
#      --res_attn \
done