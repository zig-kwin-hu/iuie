import os
import torch
import json
import random   
import numpy as np
def load_lora_weight(weight_path, device, encordec=None, selforcross=None, layer=None, module=None, AorB=None):
    weight = torch.load(weight_path, map_location=device)
    assert len(weight) > 0, 'No weight found in {}'.format(weight_path)
    weight = {k: v for k, v in weight.items() if 'lora' in k}
    assert len(weight) > 0, 'No lora weight found in {}, weight keys {}'.format(weight_path, weight.keys())
    if encordec is not None:
        weight_ = {k: v for k, v in weight.items() if '.{}.'.format(encordec) in k}
        assert len(weight_) > 0, 'No weight found for {} in {}'.format(encordec, weight.keys())
        weight = weight_
    if selforcross is not None:
        weight_ = {k: v for k, v in weight.items() if '.{}.'.format(selforcross) in k}
        assert len(weight_) > 0, 'No weight found for {} in {}'.format(selforcross, weight.keys())
        weight = weight_
    if layer is not None:
        weight_ = {k: v for k, v in weight.items() if 'block.{}.layer'.format(layer) in k}
        assert len(weight_) > 0, 'No weight found for layer {}'.format(layer)
        weight = weight_
    if module is not None:
        weight_ = {k: v for k, v in weight.items() if '.{}.'.format(module) in k}
        assert len(weight_) > 0, 'No weight found for {}'.format(module)
        weight = weight_
    if AorB is not None:
        weight_ = {k: v for k, v in weight.items() if 'lora_{}.weight'.format(AorB) in k}
        assert len(weight_) > 0, 'No weight found for {}'.format(AorB)
        weight = weight_
    
    return weight
def generate_combinations(*args):
    if len(args) == 1:
        return [[a] for a in args[0]]
    else:
        return [[a]+b for a in args[0] for b in generate_combinations(*args[1:])]
def load_all_weights(weights_dir, device, tasks, layer=23, module='v', AorB='B', encordec='encoder', selforcross='SelfAttention'):
    all_weights = []
    for task in tasks:
        task = task.lower()
        task_dir = os.path.join(weights_dir, '{}_lora'.format(task))
        datasets = os.listdir(task_dir)
        for dataset in datasets:
            if dataset == '.DS_Store':
                continue
            dataset_dir = os.path.join(task_dir, dataset)
            assert os.path.exists(os.path.join(dataset_dir, 'iuie-xxl/')), 'No weight found for {} {}'.format(task,dataset)
            dataset_dir = os.path.join(dataset_dir, 'iuie-xxl')
            checkpoint_path = os.listdir(dataset_dir)
            for checkpoint in checkpoint_path:
                if 'best_model' in checkpoint:
                    checkpoint_path = os.path.join(dataset_dir, checkpoint)
                    break
            assert os.path.exists(checkpoint_path) and 'best_model' in checkpoint_path, 'No weight found for {} {}'.format(task,dataset)
                
            checkpoint_path = os.path.join(checkpoint_path, 'adapter_model.bin')
            weight = load_lora_weight(checkpoint_path, device, encordec=encordec, selforcross=selforcross, layer=layer, module=module, AorB=AorB)
            all_weights.append({'task': task, 'dataset': dataset, 'weight': weight})
    return all_weights
def flatten_weights(weights):
    for weight in weights:
        flat_weight = []
        for k, v in weight['weight'].items():
            flat_weight.append(v.reshape(-1))
        flat_weight = torch.cat(flat_weight, dim=0).reshape(1, -1)
        weight['flat_weight'] = flat_weight
    return weights
def generate_random_cluster_embeddings(task, embedding_dim, cluster_num, dataset2id, target_dir):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    embeddings = torch.nn.Linear(embedding_dim,cluster_num).weight.data.numpy()
    unique_id2index_cluster = {}
    for dataset in dataset2id:
        path = 'data/ie_instruct_unique_id/{}/{}'.format(task, dataset)
        print('Processing {}'.format(path))
        for set_name in ['train', 'dev', 'test']:
            if not os.path.exists('{}/{}.json'.format(path, set_name)):
                continue
            with open('{}/{}.json'.format(path, set_name)) as fin:
                data = json.load(fin)
                for i in range(len(data)):
                    unique_id2index_cluster[data[i]['unique_id']] = dataset2id[dataset]
    np.save('{}/cluster_embeddings_random_{}_{}.npy'.format(target_dir, str(embedding_dim), str(cluster_num)), embeddings)
    json.dump(unique_id2index_cluster, open('{}/cluster_uid2index_random_{}_{}.json'.format(target_dir, str(embedding_dim), str(cluster_num)), 'w'), indent=2)
def generate_lora_cluster_embeddings(embeddings, dataset2id, target_dir, embedding_dim=None, dimension_reduction=None, zero_center=False, normalize=False):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if embedding_dim is not None:
        assert dimension_reduction is not None, 'Dimension reduction method not provided'
        raise NotImplementedError
    else:
        embedding_dim = embeddings.shape[1]
    cluster_num = embeddings.shape[0]
    if zero_center:
        embeddings = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    unique_id2index_cluster = {}
    for task in dataset2id:
        for dataset in dataset2id[task]:
            path = 'data/ie_instruct_unique_id/{}/{}'.format(task, dataset)
            print('Processing {}'.format(path))
            at_least_one = False
            for set_name in ['train', 'dev', 'test']:
                if not os.path.exists('{}/{}.json'.format(path, set_name)):
                    continue
                with open('{}/{}.json'.format(path, set_name)) as fin:
                    data = json.load(fin)
                    for i in range(len(data)):
                        unique_id2index_cluster[data[i]['unique_id']] = dataset2id[task][dataset]
                at_least_one = True
            assert at_least_one, 'No data found for {}'.format(path)
    np.save('{}/cluster_embeddings_lora_{}_{}_{}_{}_{}.npy'.format(target_dir, zero_center, normalize, str(embedding_dim), str(cluster_num), str(dimension_reduction)), embeddings)
    json.dump(unique_id2index_cluster, open('{}/cluster_uid2index_lora_{}_{}_{}_{}_{}.json'.format(target_dir, zero_center, normalize, str(embedding_dim), str(cluster_num), str(dimension_reduction)), 'w'), indent=2)
if __name__ == '__main__':
    weights_path = '/home/zkhu143/iuie_filtered/'
    tasks=['NER','RE','EEA','EET']
    device = 'cuda'
    layers = 23
    modules = 'q'#['q','v']
    AorB = 'A'#['A','B']
    encordec = 'decoder'#['encoder', 'decoder']
    selforcross = 'EncDecAttention'#['SelfAttention', 'EncDecAttention']
    weights = load_all_weights(weights_path, device, tasks, layer=layers, module=modules, AorB=AorB, encordec=encordec, selforcross=selforcross)
    weights = flatten_weights(weights)
    datasets = json.load(open('configs/multi_task_configs/all/train_tasks.json'))
    dataset2id = {}
    current_id = 0
    for task in tasks:
        if task not in dataset2id:
            dataset2id[task] = {}
        for dataset in datasets[task]:
            dataset2id[task][dataset['dataset name']] = current_id
            current_id += 1
    target_dir = 'data/ie_instruct_unique_id/cluster_embeddings/multi_task/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    id2dataset = {}
    for task in dataset2id:
        for dataset in dataset2id[task]:
            id2dataset[dataset2id[task][dataset]] = {'task':task,'dataset':dataset}
    dataset2weight = {task:{} for task in tasks}
    for weight in weights:
        dataset = weight['dataset']
        task = weight['task'].upper()
        if dataset not in dataset2weight[task] and (weight['task'] in tasks or weight['task'] in [t.lower() for t in tasks]):
            dataset2weight[task][dataset] = weight['flat_weight']
    lora_embeddings = []
    for id in range(len(id2dataset)):
        dataset = id2dataset[id]['dataset']
        task = id2dataset[id]['task']
        if dataset not in dataset2weight[task]:
            dataset = '_'.join(dataset.split(' '))
        if dataset not in dataset2weight[task]:
            dataset = dataset.split('_sample')[0]
        if dataset not in dataset2weight[task]:
            dataset = dataset.lower()
        if dataset not in dataset2weight[task] and dataset == 'ace05':
            dataset = 'ace'
        assert dataset in dataset2weight[task], 'No weight found for {}\n existing datasets are {}'.format(dataset, dataset2weight[task].keys())
        lora_embeddings.append(dataset2weight[task][dataset])
    lora_embeddings = torch.cat(lora_embeddings, dim=0).cpu().float().numpy()
    print('Lora embeddings shape', lora_embeddings.shape)

    generate_lora_cluster_embeddings(lora_embeddings, dataset2id, target_dir, zero_center=True, normalize=False)
    