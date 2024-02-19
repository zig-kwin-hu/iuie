import os
import json
import random
import numpy as np
import torch
def generate_random_sentence_embeddings(source_dir, embedding_dim):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    task_dirs = os.listdir(source_dir)
    for task_dir in task_dirs:
        dataset_dirs = os.listdir('{}/{}'.format(source_dir, task_dir))
        if task_dir in ['NER_LLM']:
            continue
        for dataset_dir in dataset_dirs:
            print('Processing {}/{}'.format(task_dir, dataset_dir))
            sets = ['train', 'dev', 'test']
            unique_id2index_sentence = {}
            embeddings = []
            for set_name in sets:
                if not os.path.exists('{}/{}/{}/{}.json'.format(source_dir, task_dir, dataset_dir, set_name)):
                    continue
                with open('{}/{}/{}/{}.json'.format(source_dir, task_dir, dataset_dir, set_name)) as fin:
                    data = json.load(fin)
                for i in range(len(data)):
                    generated = torch.nn.Linear(embedding_dim,1).weight.data.numpy().reshape(-1)
                    embeddings.append(generated)
                    unique_id2index_sentence[data[i]['unique_id']] = len(embeddings) - 1
            target_path = '{}/{}/{}/sentence_embedding_random_{}.npy'.format(source_dir, task_dir, dataset_dir, str(embedding_dim))
            np.save(target_path, np.array(embeddings))
            print('Embeddings saved to {}'.format(target_path))
            
            json.dump(unique_id2index_sentence, open('{}/{}/{}/unique_id2index_sentence_random_{}.json'.format(source_dir, task_dir, dataset_dir, embedding_dim), 'w'), indent=2)
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
if __name__ == '__main__':
    prefix = 'data/ie_instruct_unique_id/RE/'
    task='RE'
    datasets = json.load(open('configs/re_configs/all/dev_tasks.json'))[task]
    dataset2id = {}
    for dataset in datasets:
        dataset2id[dataset['dataset name']] = len(dataset2id)
    target_dir = 'data/ie_instruct_unique_id/cluster_embeddings/re/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    generate_random_cluster_embeddings(task, 4096, len(dataset2id), dataset2id, target_dir)
