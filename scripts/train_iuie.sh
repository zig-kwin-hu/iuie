#!/bin/bash
declare -A epoch_map
epoch_map=([with_sentence_iuie_mean_of_encoder]=30 [NYT11_NYT]=10 [semval-RE]=10 [NYT11]=10 [SciERC]=20 [NYT11_semval-RE]=20 [NYT11_SciERC]=20 [ADE_corpus-1500]=25 [ADE_NYT11]=25 [ADE_SciERC]=25 [ADE_semval-RE]=25 [semval-RE_SciERC]=20 [SciERC_NYT11]=20 [SciERC_ADE]=25 [4combined]=20 [semval-RE_ADE]=25 [semval-RE_NYT11]=20 [SciERC_semval-RE]=20)

# declare -A TASK2DATASETS=([re]="conll04 SciERC NYT11 semval-RE ADE_corpus-1500" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="CoNLL_2003 ACE_2004 ACE_2005")
# DONE : [ner] = ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7_sample_15000 WikiANN_en WikiNeural
#declare -A TASK2DATASETS=([re]="ADE_corpus NYT11_sample_30000 New-York-Times-RE_sample_30000 semval-RE conll04 GIDS SciERC kbp37" [eet]="ace phee casie" [eea]="ace phee casie" [ner]="ACE_2004 ACE_2005 AnatEM bc2gm bc4chemd bc5cdr Broad_Tweet_Corpus CoNLL_2003 FabNER FindVehicle GENIA_NER HarveyNER mit-movie mit-restaurant MultiNERD ncbi Ontonotes_sample_30000 PolyglotNER TweetNER7 WikiANN_en WikiNeural")
declare -A TASK2DATASETS=([ner]="plo_all" [re]="SciERC" [with_sentence_iuie_mean_of_encoder]="0_2" [ner_cluster]="ACE_2004_ACE_2005" [re_cluster]="NYT11_NYT" [eet]="ace phee casie" [eea]="ace phee casie")

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=./huggingface

port=$(shuf -i25000-30000 -n1)

model_name_or_path=ZWK/InstructUIE
lora_r=16
lora_alpha=16
add_name=True
#--per_device_train_batch_size 10 \
#--gradient_accumulation_steps 3 \
# model_name_or_path=google/flan-t5-xxl

# for TASK in re ner eet eea 
for TASK_CONFIG in re
do
    for DATASET_CONFIG in ${TASK2DATASETS[${TASK_CONFIG}]}
    do
        if [[ "$DATASET_CONFIG" =~ ^(plo_all|re_all|disease)$ ]]; then
            over_sample=True
        else
            over_sample=False
        fi
        if [[ "${add_name}" == "True" ]]; then
            output_dir="output/${TASK_CONFIG}_lora/${DATASET_CONFIG}/iuie-xxl_addname_${lora_r}"
        else
            output_dir="output/${TASK_CONFIG}_lora/${DATASET_CONFIG}/iuie-xxl_${lora_r}"
        fi
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_uie.py \
        --do_train \
        --do_eval \
        --do_predict \
        --num_beams 1 \
        --repetition_penalty 1.0 \
        --predict_with_generate \
        --model_name_or_path ${model_name_or_path} \
        --data_dir ./data/ie_instruct \
        --task_config_dir ./configs/${TASK_CONFIG}_configs/${DATASET_CONFIG} \
        --instruction_file ./configs/instruction_config.json \
        --prompt_file ./prompts/instructUIE.json \
        --instruction_strategy multiple \
        --min_negative_labels -1 \
        --min_positive_labels -1 \
        --output_dir "${output_dir}" \
        --input_record_file iuie.record \
        --per_device_train_batch_size 5 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 6 \
        --learning_rate 5e-05 \
        --num_train_epochs 4 \
        --run_name ${model_name_or_path}-${TASK_CONFIG}-${DATASET_CONFIG} \
        --max_source_length 256 \
        --max_target_length 50 \
        --generation_max_length 50 \
        --max_num_instances_per_task 20000 \
        --max_num_instances_per_eval_task 500 \
        --max_num_instances_per_predict_task 500 \
        --add_task_name ${add_name} \
        --add_dataset_name ${add_name} \
        --num_examples 0 \
        --overwrite_cache \
        --lr_scheduler_type constant \
        --warmup_step 0 \
        --logging_strategy steps \
        --logging_steps 100 \
        --cache_dir ./huggingface \
        --ddp_find_unused_parameters False \
        --save_total_limit 50 \
        --over_sampling ${over_sample} \
        --bf16 True \
        --load_best_model_at_end True \
        --metric_for_best_model eval_f1 \
        --early_stopping_patience 5 \
        --only_save_best_model True \
        --lora_target_modules q,v \
        --lora_r ${lora_r} \
        --lora_alpha ${lora_alpha} \
        --use_test_as_eval \
        --group_by_length \
        --save_lora_weights_only \
        --predict_each_dataset_with_best False \
        --auto_find_best_lora_checkpoint False \
        --save_strategy steps \
        --save_steps 50 \
        --evaluation_strategy steps \
        --eval_steps 50 \
        #--moe_lora True \
        #--resume_from_checkpoint /home/zkhu143/iuie/output/ner_lora/plo_all/iuie-xxl/checkpoint-30 \
        #--evaluation_strategy epoch \
        #--save_strategy epoch \
        #--evaluation_strategy epoch \
        #--save_strategy epoch \
        #--overwrite_output_dir \
    done
done
for TASK_CONFIG in ner
do
    for DATASET_CONFIG in ${TASK2DATASETS[${TASK_CONFIG}]}
    do  
        if [[ "$DATASET_CONFIG" =~ ^(plo_all|re_all|disease)$ ]]; then
            over_sample=True
        else
            over_sample=False
        fi
        if [[ "${add_name}" == "True" ]]; then
            output_dir="output/${TASK_CONFIG}_lora/${DATASET_CONFIG}/iuie-xxl_addname_${lora_r}"
        else
            output_dir="output/${TASK_CONFIG}_lora/${DATASET_CONFIG}/iuie-xxl_${lora_r}"
        fi
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_uie.py \
        --do_predict \
        --num_beams 1 \
        --repetition_penalty 1.0 \
        --predict_with_generate \
        --model_name_or_path ${model_name_or_path} \
        --data_dir ./data/ie_instruct \
        --task_config_dir ./configs/${TASK_CONFIG}_configs/${DATASET_CONFIG} \
        --instruction_file ./configs/instruction_config.json \
        --prompt_file ./prompts/instructUIE.json \
        --instruction_strategy multiple \
        --min_negative_labels -1 \
        --min_positive_labels -1 \
        --output_dir "${output_dir}" \
        --input_record_file iuie.record \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 3 \
        --learning_rate 5e-05 \
        --num_train_epochs 10 \
        --run_name ${model_name_or_path}-${TASK_CONFIG}-${DATASET_CONFIG} \
        --max_source_length 256 \
        --max_target_length 50 \
        --generation_max_length 50 \
        --max_num_instances_per_task 20000 \
        --max_num_instances_per_eval_task -1 \
        --max_num_instances_per_predict_task -1 \
        --add_task_name ${add_name} \
        --add_dataset_name ${add_name} \
        --num_examples 0 \
        --overwrite_cache \
        --lr_scheduler_type constant \
        --warmup_step 0 \
        --logging_strategy steps \
        --logging_steps 100 \
        --cache_dir ./huggingface \
        --ddp_find_unused_parameters False \
        --save_total_limit 30 \
        --over_sampling  ${over_sample}\
        --bf16 True \
        --load_best_model_at_end True \
        --metric_for_best_model eval_f1 \
        --only_save_best_model True \
        --lora_target_modules q,v \
        --lora_r ${lora_r} \
        --lora_alpha ${lora_alpha} \
        --use_test_as_eval \
        --group_by_length \
        --save_lora_weights_only \
        --predict_each_dataset_with_best False \
        --auto_find_best_lora_checkpoint True \
        --save_strategy steps \
        --save_steps 500 \
        --evaluation_strategy steps \
        --eval_steps 500 \
        #--resume_from_checkpoint /home/zkhu143/iuie/output/ner_lora/plo_all/iuie-xxl/checkpoint-30 \
        #--evaluation_strategy epoch \
        #--save_strategy epoch \
        #--evaluation_strategy epoch \
        #--save_strategy epoch \
        #--overwrite_output_dir \
    done
done
#iuie
#deepspeed
    #bf16
        #bsz 3 grad_acc 1 lora 16 iters 1140 3.60s/it 1:08:14  max 16208 MiB
    #naive load
        #bsz=3 grad_acc 1 lora 16 iters 1140 6.88s/it 2:00:14  max 22416 MiB
#no deepspeed
    #bf16
        # bsz 3 grad_acc 1 lora 16 iters 4560 3.23it/s 27:14  max 10446 MiB
        # bsz 6 grad_acc 1 lora 16 iters 2280 1.73it/s 20:14  max 14324 MiB
        # bsz 6 grad_acc 3 lora 16 iters 760  1.70s/1t 20:00  max 14324 MiB
        # bsz 10 grad_acc 3 lora 16 iters 450  2.65s/1t 19:16  max 19660 MiB
    #naive load
        # bsz 6 grad_acc 1 lora 16 iters 2280 1.95s/it 1:12:55  max 21036 MiB
#{"eval": {"eval_loss": 0.25069040060043335, "eval_exact_match": 61.3477, "eval_rouge1": 78.638, "eval_rougeL": 78.5585, "eval_F1_for_PolyglotNER_sample_20000": 0.583, "eval_F1_for_Broad_Tweet_Corpus": 0.808, "eval_F1_for_WikiANN_en": 0.6117, "eval_F1_for_WikiNeural_sample_20000": 0.8774, "eval_f1": 0.72, "eval_exact_match_for_PolyglotNER_sample_20000": 45.28, "eval_rouge1_for_PolyglotNER_sample_20000": 67.1884, "eval_rougeL_for_PolyglotNER_sample_20000": 67.0591, "eval_exact_match_for_Broad_Tweet_Corpus": 67.3, "eval_rouge1_for_Broad_Tweet_Corpus": 85.1958, "eval_rougeL_for_Broad_Tweet_Corpus": 85.0294, "eval_exact_match_for_WikiANN_en": 51.92, "eval_rouge1_for_WikiANN_en": 75.1895, "eval_rougeL_for_WikiANN_en": 75.1091, "eval_exact_match_for_WikiNeural_sample_20000": 82.3058, "eval_rouge1_for_WikiNeural_sample_20000": 90.3536, "eval_rougeL_for_WikiNeural_sample_20000": 90.3328, "eval_gen_len": 10.6674, "eval_global_step": 15, "eval_runtime": 8443.0731, "eval_samples_per_second": 3.979, "eval_steps_per_second": 0.249, "epoch": 0.01}}
#{"eval": {"eval_loss": 0.21326033771038055, "eval_exact_match": 62.0532, "eval_rouge1": 78.4653, "eval_rougeL": 78.397, "eval_F1_for_PolyglotNER_sample_20000": 0.6081, "eval_F1_for_Broad_Tweet_Corpus": 0.7677, "eval_F1_for_WikiANN_en": 0.6086, "eval_F1_for_WikiNeural_sample_20000": 0.8785, "eval_f1": 0.7158, "eval_exact_match_for_PolyglotNER_sample_20000": 48.62, "eval_rouge1_for_PolyglotNER_sample_20000": 69.7729, "eval_rougeL_for_PolyglotNER_sample_20000": 69.6461, "eval_exact_match_for_Broad_Tweet_Corpus": 62.2, "eval_rouge1_for_Broad_Tweet_Corpus": 79.827, "eval_rougeL_for_Broad_Tweet_Corpus": 79.6909, "eval_exact_match_for_WikiANN_en": 51.43, "eval_rouge1_for_WikiANN_en": 73.3281, "eval_rougeL_for_WikiANN_en": 73.2625, "eval_exact_match_for_WikiNeural_sample_20000": 82.7714, "eval_rouge1_for_WikiNeural_sample_20000": 90.1557, "eval_rougeL_for_WikiNeural_sample_20000": 90.1471, "eval_gen_len": 10.2541, "eval_global_step": 30, "eval_runtime": 8285.121, "eval_samples_per_second": 4.055, "eval_steps_per_second": 0.253, "epoch": 0.01}}
