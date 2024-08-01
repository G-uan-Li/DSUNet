#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# -------------------Training Shell Script--------------------
# 三个学习率，所以要训练 3 次 
if true; then

  transformer_type=bert
  # transformer_type=roberta 

  channel_type=context-based
  # channel_type=grouped-bilinear
  # channel_type=similarity-based
  if [[ $transformer_type == bert ]]; then
    bs=4
    bl=3e-5
    # 可以修改学习率，调参
    uls=( 3e-4 4e-4 5e-4)
    accum=1
    for ul in ${uls[@]}
    do
    python -u ./train_balanceloss.py --data_dir ./dataset/docred \
    --channel_type $channel_type \
    --bert_lr $bl \
    --transformer_type $transformer_type \
    --model_name_or_path /root/autodl-tmp/bert-base-cased \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 3 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --num_train_epochs 30 \
    --seed 66 \
    --num_class 97 \
    --save_path ./checkpoint/docred/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_bert_accum${accum}_lr${ul}_type_${channel_type}_bs_${bs}.log
    done


  elif [[ $transformer_type == roberta ]]; then
    type=context-based
    
    bs=2
    bls=(3e-5)
    ul=4e-4
    accum=2
    for bl in ${bls[@]}
    do
    python -u ./train_balanceloss.py --data_dir ./dataset/docred \
    --channel_type $channel_type \
    --bert_lr $bl \
    --transformer_type $transformer_type \
    --model_name_or_path /root/autodl-tmp/roberta-large \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 4 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30 \
    --seed 111 \
    --num_class 97 \
    --save_path ./checkpoint/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    done
  fi
fi
