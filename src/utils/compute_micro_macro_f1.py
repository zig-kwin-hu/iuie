import copy
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from compute_metrics import compute_f1, compute_metrics, compute_grouped_metrics
import numpy as np
def compute_micro_macro_f1_metrics(dataset, preds, save_prefix='dev'):
    decoded_preds = preds
    #references = [e["Instance"]["label"] for e in dataset]
    #result = compute_metrics(predictions=decoded_preds, references=references)
    result = {}
    result_per_task = compute_f1(dataset, decoded_preds)
    result.update(result_per_task)
    #result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                #   groups=categories)
    #result.update(result_per_category)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
if __name__ == '__main__':
    prediction_path = '/home/zkhu143/iuie/output_ssd2/zeroshot_moelora/all/no_cluster_embedding_for_gate/InstructUIE_16_TopKGate_8_2_router_z_True_False_False_False_multitask_all/test_eval_predictions.jsonl'
    lines = open(prediction_path).readlines()
    print('len(lines)', len(lines))
    dataset = [json.loads(l) for l in lines]
    preds = [e["Prediction"] for e in dataset]
    task2dataset = {}
    for record in dataset:
        task = record['Task']
        if task not in task2dataset:
            task2dataset[task] = []
        dataset_name = record['Dataset']
        if dataset_name not in task2dataset[task]:
            task2dataset[task].append(dataset_name)
    dataset2task = {dataset_name:task for task in task2dataset for dataset_name in task2dataset[task]}
    result = compute_micro_macro_f1_metrics(dataset, preds)
    print(result)
    renamed_dataset = copy.deepcopy(dataset)
    for record in renamed_dataset:
        if record['Dataset'] in dataset2task:
            record['Dataset'] = dataset2task[record['Dataset']]
    result = compute_micro_macro_f1_metrics(renamed_dataset, preds)
    print(result)

