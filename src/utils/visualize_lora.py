from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
def reverse_upper_lower(s):
    return s.lower() if s.isupper() else s.upper()
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
def load_all_weights(weights_dir, device, layer=23, module='v', AorB='B', encordec='encoder', selforcross='SelfAttention'):
    all_weights = []
    for task in ['eea','eet','ner', 're']:
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
def transform_weights(weights, zero_center=True, unit_variance=True):
    all_weights = [weight['flat_weight'] for weight in weights]
    all_weights = torch.cat(all_weights, dim=0)
    if zero_center:
        all_weights = all_weights - all_weights.mean(dim=0)
    if unit_variance:
        all_weights = all_weights / all_weights.std(dim=0)
    for i, weight in enumerate(weights):
        weight['flat_weight'] = all_weights[i]
    return weights
def tsne(weights):
    all_weights = [weight['flat_weight'].float().cpu().numpy() for weight in weights]
    all_weights = np.concatenate(all_weights, axis=0)
    print(all_weights.shape)
    #there are 27 samples, need to decide an appropriate perplexity
    perplexity = 5
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    all_weights = tsne.fit_transform(all_weights)
    for i, weight in enumerate(weights):
        weight['tsne'] = all_weights[i]
    return weights
def visualize(weights, color_map, marker_map, save_path, datasets=None, bounds=None):
    #Each weight is a dictionary with keys: task, dataset, weight, flat_weight, tsne,
    #All the weights with the same task has the same color, and all the weights with the same dataset has the same marker
    #The x-axis is the first dimension of the tsne, and the y-axis is the second dimension of the tsne

    for weight in weights:
        if datasets is None or weight['dataset'] in datasets:
            plt.scatter(weight['tsne'][0], weight['tsne'][1], c=color_map[weight['task']], marker=marker_map[weight['dataset']], label=weight['dataset'])
    #run on a server, so no display, save the figure
    if bounds is not None:
        plt.xlim(bounds[0]-5, bounds[1]+5)
        plt.ylim(bounds[2]-5, bounds[3]+5)
    #make sure the size of the figure is the same as the bound
        
    plt.savefig(save_path)
    plt.close()
if __name__ == '__main__':
    weights_path = '/home/zkhu143/iuie_filtered/'
    device = 'cuda'
    layers = [23]
    modules = ['q']#['q','v']
    AorB = ['A']#['A','B']
    encordec = ['decoder']#['encoder', 'decoder']
    selforcross = ['EncDecAttention']#['SelfAttention', 'EncDecAttention']
    combinations = generate_combinations(layers, modules, AorB, encordec, selforcross)
    for combination in combinations:
        print(combination)
        if 'encoder' in combination and 'EncDecAttention' in combination:
            continue
        all_weights = load_all_weights(weights_path, device, layer=combination[0], module=combination[1], AorB=combination[2], encordec=combination[3], selforcross=combination[4])
        all_weights = flatten_weights(all_weights)
        all_weights = transform_weights(all_weights, zero_center=True, unit_variance=True)
        all_weights = tsne(all_weights)
        #get upperbound and lowerbound of the tsne coordinates
        minx = min([weight['tsne'][0] for weight in all_weights])
        maxx = max([weight['tsne'][0] for weight in all_weights])
        miny = min([weight['tsne'][1] for weight in all_weights])
        maxy = max([weight['tsne'][1] for weight in all_weights])
        print(minx, maxx, miny, maxy)
        bounds = [minx, maxx, miny, maxy]
        color_map = {'eea': 'r', 'eet': 'g', 'ner': 'b', 're': 'y'}
        datasets = set()
        for weight in all_weights:
            datasets.add(weight['dataset'])
        marker_map = None
        save_path = 'output_ssd2/visualization/weights_tsne_plo_{}_{}_{}_{}_{}.png'.format(combination[0], combination[1], combination[2], combination[3], combination[4])
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))   
        
        marker_map = {dataset: 'o' for dataset in datasets}
        star_datasets = ['WikiNeural','Broad_Tweet_Corpus','WikiANN_en','PolyglotNER'] 
        for dataset in star_datasets:
            marker_map[dataset] = '1'
        for weight in all_weights:
            if weight['dataset'] in star_datasets:
                print(weight['dataset'], weight['tsne'])
        visualize(all_weights, color_map, marker_map, save_path, datasets=None, bounds=bounds)
    
