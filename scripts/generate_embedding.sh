#!/bin/bash
declare -A epoch_map
epoch_map=([NYT11_semval-RE]=20 [NYT11_SciERC]=20 [ADE_corpus-1500]=25 [ADE_NYT11]=25 [ADE_SciERC]=25 [ADE_semval-RE]=25 [semval-RE_SciERC]=20 [SciERC_NYT11]=20 [SciERC_ADE]=25 [4combined]=20 [semval-RE_ADE]=25 [semval-RE_NYT11]=20 [SciERC_semval-RE]=20)

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=ZWK/InstructUIE


for TASK_CONFIG in EMBED_INSTRUCTION_with_sentence
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/generate_embedding.py \
    --do_predict \
    --model_name_or_path ${model_name_or_path} \
    --data_dir ./data/ie_instruct \
    --task_config_dir ./configs/embed_configs/${TASK_CONFIG} \
    --instruction_file ./configs/instruction_config.json \
    --prompt_file ./prompts/instructUIE.json \
    --instruction_strategy single \
    --min_negative_labels -1 \
    --min_positive_labels -1 \
    --output_dir ./output/${TASK_CONFIG}/iuie_mean_of_encoder\
    --input_record_file iuie.record \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-05 \
    --num_train_epochs 0 \
    --run_name ${model_name_or_path}-EMBED-${TASK_CONFIG} \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_num_instances_per_task 4000 \
    --max_num_instances_per_eval_task 200 \
    --add_task_name False \
    --add_dataset_name False \
    --num_examples 0 \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --bf16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --cache_dir ./huggingface \
    --ddp_find_unused_parameters False \
    --save_total_limit 1 \
    --over_sampling False \
    --load_best_model_at_end False\
    --metric_for_best_model eval_f1 \
    --only_save_best_model True \
    --predict_each_dataset_with_best False \
    --overwrite_cache \
    --embedding_prompt iota \
    --embedding_type mean_of_encoder \
    --predict_with_generate \
    --overwrite_cache \
    #--lora_target_modules q,v \
    #--lora_r 16 \
    #--test_with_eval \
    #--save_lora_weights_only \
    #--overwrite_output_dir \
done

