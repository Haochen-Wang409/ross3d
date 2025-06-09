#!/bin/bash

MID_RUN_NAME="llava-video-qwen2-7b-ross3d"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# bs=256, lr=1e-5
set -x

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=20409 \
    \
    ross3d/train/train_3d.py \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --view_mask_ratio 0.25 \
    --view_mask_prob 0.25 \
    \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmms-lab/LLaVA-Video-7B-Qwen2 \
    --pretrain_mm_inv_adapter ./checkpoints/mm_inv_projector.bin \
    --version qwen_1_5 \
    --data_path scripts/3d/train/video3dllm_223k.yaml \
    --image_folder data \
    --video_folder data \
    --embodiedscan_folder data/embodiedscan/ \
    --mm_tunable_parts "mm_mlp_adapter,mm_language_model,mm_inv_adapter" \
    --vision_tower /data/llm_model_zoo/siglip-so400m-patch14-384 \
    --ross_multi_task True \
    --mm_pixel_decoder ./checkpoints/FLUX.1-dev \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --save_only_model \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 32 \
    --report_to wandb --run_name $MID_RUN_NAME
