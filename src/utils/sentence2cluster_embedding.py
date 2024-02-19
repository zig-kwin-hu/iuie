import numpy as np
import os
import json
def aggregate_sentence_embeddings(sentence_embeddings, aggregation_method):
    if aggregation_method == 'mean':
        return np.mean(sentence_embeddings, axis=0)
    elif aggregation_method == 'max':
        return np.max(sentence_embeddings, axis=0)
    elif aggregation_method == 'min':
        return np.min(sentence_embeddings, axis=0)
    elif aggregation_method == 'sum':
        return np.sum(sentence_embeddings, axis=0)
    elif aggregation_method == 'first':
        return sentence_embeddings[0]
    elif aggregation_method == 'last':
        return sentence_embeddings[-1]
    else:
        raise NotImplementedError
def expand_cluster_embeddings_from_sentence_embeddings(cluster_embeddings, cluster_uid2index, output_dir, task, dataset_name, suffix):
    data_dir = 'data/ie_instruct_unique_id/{}/{}'.format(task.upper(), dataset_name)
    sets = ['train', 'dev', 'test']
    for set_name in sets:
        if not os.path.exists('{}/{}.json'.format(data_dir, set_name)):
            continue
        with open('{}/{}.json'.format(data_dir, set_name)) as fin:
            data = json.load(fin)
            for i in range(len(data)):
                
                if data[i]['unique_id'] in cluster_uid2index:
                    print('Duplicated unique_id: {}'.format(data[i]['unique_id']))
                    exit(0)
                cluster_uid2index[data[i]['unique_id']] = len(cluster_embeddings)
    new_embeddings = np.load(os.path.join(output_dir, '{}/{}/embeddings_{}.npy'.format(task, dataset_name, suffix)))
    aggregated_embeddings = aggregate_sentence_embeddings(new_embeddings, 'mean')
    cluster_embeddings.append(aggregated_embeddings)
    return cluster_embeddings, cluster_uid2index, 
if __name__ == '__main__':
    cluster_embeddings = []
    cluster_uid2index = {}
    output_dir = 'output_ssd2/embedding/'
    tasks = ['ner']
    suffix = 'InstructUIE_iota_mean_of_encoder_eval_0'
    for task in tasks:
        dataset_names = os.listdir(os.path.join(output_dir, task))
        for dataset_name in dataset_names:
            if not os.path.exists(os.path.join(output_dir,'{}/{}/embeddings_{}.npy'.format(task, dataset_name, suffix))):
                continue
            new_embeddings = np.load(os.path.join(output_dir, '{}/{}/embeddings_{}.npy'.format(task, dataset_name, suffix)))
            cluster_embeddings, cluster_uid2index = expand_cluster_embeddings_from_sentence_embeddings(
                cluster_embeddings, cluster_uid2index, output_dir, task, dataset_name, suffix)
    print('Total number of clusters: {}'.format(len(cluster_embeddings)))
    print('total number of unique ids: {}'.format(len(cluster_uid2index)))
    target_dir = 'data/ie_instruct_unique_id/cluster_embeddings/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    np.save(os.path.join(target_dir, 'cluster_embeddings_{}.npy'.format(suffix)), np.array(cluster_embeddings))
    json.dump(cluster_uid2index, open(os.path.join(target_dir, 'cluster_uid2index_{}.json'.format(suffix)), 'w'), indent=2)

    