if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/MOR" ]; then
    mkdir ./logs/MOR
fi

export CUDA_VISIBLE_DEVICES=2

seq_len=96
label_len=48
model_name=MSRNet

pred_len=96
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path MOR.csv \
    --model_id MOR'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data MOR \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --target_num 3 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 #>logs/MOR/$model_name'_'MOR_$seq_len'_'$pred_len.log

pred_len=192
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path MOR.csv \
    --model_id MOR'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data MOR \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --target_num 3 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 #>logs/MOR/$model_name'_'MOR_$seq_len'_'$pred_len.log


pred_len=336
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path MOR.csv \
    --model_id MOR'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data MOR \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --target_num 3 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 #>logs/MOR/$model_name'_'MOR_$seq_len'_'$pred_len.log


pred_len=720
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path MOR.csv \
    --model_id MOR'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data MOR \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --target_num 3 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 #>logs/MOR/$model_name'_'MOR_$seq_len'_'$pred_len.log