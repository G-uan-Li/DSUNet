# 设置CUDA可见的设备为0
export CUDA_VISIBLE_DEVICES=0

# -------------------Training Shell Script--------------------
# 如果为真
if true; then
  transformer_type=roberta 
  
  
  channel_type=context-based
# tychannel_type=grouped-bilinear
  # channel_type=similarity-based

  # 如果transformer_type为roberta
  if [[ $transformer_type == roberta ]]; then
    bs=2
    bls=3e-5
    ul=4e-4 
    accum=2

    # 循环遍历bls数组中的值
    for bl in ${bls[@]}

    do
      # 运行train_balanceloss.py脚本
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
      --num_labels 3 \
      --learning_rate $ul \
      --max_grad_norm 1.0 \
      --warmup_ratio 0.06 \
      --num_train_epochs 30 \
      --seed 66 \
      --num_class 97 \
      --save_path ./checkpoint/docred/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
      --log_dir ./logs/docred/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    done
  fi
fi