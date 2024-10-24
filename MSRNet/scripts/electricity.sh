if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/electricity" ]; then
    mkdir ./logs/electricity
fi

export CUDA_VISIBLE_DEVICES=3

seq_len=96
label_len=48
model_name=MSRNet

pred_len=96
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
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
    --target_num 8 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log

pred_len=192
python -u run_longExp.py \
   --is_training 1 \
   --root_path ./dataset/ \
   --data_path electricity.csv \
    --model_id electricity'_'$seq_len'_'$pred_len \
   --model $model_name \
   --data custom \
   --features M \
   --freq h \
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
   --target_num 8 \
   --des 'Exp' \
   --d_model 32 \
   --d_ff 64 \
   --top_k 3 \
   --conv_channel 32 \
   --skip_channel 32 \
   --batch_size 32 \
   --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log

pred_len=336
python -u run_longExp.py \
   --is_training 1 \
   --root_path ./dataset/ \
   --data_path electricity.csv \
   --model_id electricity'_'$seq_len'_'$pred_len \
   --model $model_name \
   --data custom \
   --features M \
   --freq h \
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
   --target_num 8 \
   --des 'Exp' \
   --d_model 32 \
   --d_ff 64 \
   --top_k 3 \
   --conv_channel 32 \
   --skip_channel 32 \
   --batch_size 32 \
   --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log


pred_len=720
python -u run_longExp.py \
   --is_training 1 \
   --root_path ./dataset/ \
   --data_path electricity.csv \
   --model_id electricity'_'$seq_len'_'$pred_len \
   --model $model_name \
   --data custom \
   --features M \
   --freq h \
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
   --target_num 8 \
   --des 'Exp' \
   --d_model 32 \
   --d_ff 64 \
   --top_k 3 \
   --conv_channel 32 \
   --skip_channel 32 \
   --node_dim 100 \
   --batch_size 32 \
   --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log