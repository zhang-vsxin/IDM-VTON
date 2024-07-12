export MODEL_NAME="../models"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch train_inf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --center_crop  \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4  \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpointing_steps 1000 \
  --output_dir="output" \
  --dataroot="./data"  \
  --width 384  \
  --height 512  \
  --pretrained_nonfreeze_model_name_or_path="../models"  \
  --validation_steps 200  \
  --enable_xformers_memory_efficient_attention  \
  --seed 42 

