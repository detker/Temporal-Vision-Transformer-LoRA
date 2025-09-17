#!/bin/bash

KAGGLE_DATASET="matthewjansen/ucf101-action-recognition"
WORKING_DIR="work_dir"
EXPERIMENT_NAME="ViT_LoRA_Temporal_UCF101_Training"
DATA_FOLDER="data"
TARGET_DIR="${WORKING_DIR}/${EXPERIMENT_NAME}/${DATA_FOLDER}"

pip install kaggle unzip

mkdir -p "$TARGET_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR"

ZIP_FILE=$(ls "$TARGET_DIR"/*.zip | head -n 1)
if [ -f "$ZIP_FILE" ]; then
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

    pip install -r requirements.txt

    accelerate launch train.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --working_directory "${WORKING_DIR}" \
        --path_to_data "${DATA_FOLDER}" \
        --checkpoint_dir 'checkpoints' \
        --hf_model_name 'google/vit-base-patch16-224' \
        --lora_rank 4 \
        --lora_alpha 8 \
        --lora_use_rslora \
        --lora_dropout 0.1 \
        --lora_bias 'lora_only' \
        --lora_target_modules 'patch_embd,q,k,v,linear1,linear2' \
        --lora_exclude_modules 'head' \
        --max_grad_norm 1.0 \
        --per_gpu_batch_size 144 \
        --gradient_accumulation_steps 8 \
        --warmup_epochs 3 \
        --epochs 150 \
        --save_checkpoint_interval 1 \
        --learning_rate 3e-5 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --max_no_of_checkpoints 3 \
        --img_size 224 \
        --num_workers 12 \
        --top_k 5 \
        --n_frames 18 \
        --custom_weight_init \
        --log_wandb \
#        --resume_from_checkpoint 'checkpoint_126'

else
    echo "No zip file found in $TARGET_DIR."
fi




