import json
import numpy as np
import copy
#record = {'attention_mode':attention_mode, 'stack_mode':stack_mode, 'unique_id':unique_id, 'gate_load':gate_load_i, 'layer_index':layer_index}
#gate_load_i = {'q':{},'k':{},'v':{}}
#q = {'top1_ratio':top1_ratio, 'scores':scores}
def unevenness_measure(v):
    N = len(v)
    if N <= 1:
        return 0  # For a vector with a single element, the unevenness is 0.
    
    # Calculate the variance of the distribution
    variance = np.var(v)
    
    # Maximum variance occurs for a one-hot vector, which we can calculate
    max_variance = (1 - 1/N) * (1/N)
    
    # Normalize the variance by the maximum possible variance
    normalized_variance = variance / max_variance
    
    return normalized_variance
def process_gate_loads(gate_path, verbose=False):
    lines = open(gate_path).readlines()
    dataset2gate_record = {}
    gate_record_template = {
        'encoder':{'self_attention':{}, 'cross_attention':{}},
        'decoder':{'self_attention':{}, 'cross_attention':{}},
    }
    for l in lines:
        j = json.loads(l)
        try:
            unique_id = j['unique_id']
        except:
            print(l, 'does not have unique_id')
            exit(0)
        try:
            splited = unique_id.split('_')
            task = splited[0]
            dataset = '_'.join(splited[1:-2])
            set_name = splited[-2]
            index = splited[-1]
        except:
            print('unknown unique_id format:', unique_id)
            exit(0)
        if task+'_'+dataset not in dataset2gate_record:
            dataset2gate_record[task+'_'+dataset] = copy.deepcopy(gate_record_template)
        gate_record = dataset2gate_record[task+'_'+dataset]
        record_layers = gate_record[j['stack_mode']][j['attention_mode']]
        gate_load = j['gate_load']
        layer_index = j['layer_index']

        for module in gate_load:
            if gate_load[module] is None:
                continue
            gate_num =len(gate_load[module]['top1_ratio'])
            if module not in record_layers:
                record_layers[module] = {}
            if layer_index not in record_layers[module]:
                record_layers[module][layer_index] = {'top1_ratio':np.zeros(gate_num),
                'scores':np.zeros(gate_num),'total_num':0}
            record_layers[module][layer_index]['top1_ratio'] += np.array(gate_load[module]['top1_ratio'])
            record_layers[module][layer_index]['scores'] += np.array(gate_load[module]['scores'])
            record_layers[module][layer_index]['total_num'] += 1
    for dataset in dataset2gate_record:
        gate_record = dataset2gate_record[dataset]
        for stack_mode in gate_record:
            for attention_mode in gate_record[stack_mode]:
                for module in gate_record[stack_mode][attention_mode]:
                    record_layers = gate_record[stack_mode][attention_mode][module]
                    #print in a readable format, with column and row names, with the same space for each name and number
                    if verbose:
                        print('dataset:', dataset, 'stack_mode:', stack_mode, 'attention_mode:', attention_mode, 'module:', module) 

                    for l in range(len(record_layers)):
                        record_layer = record_layers[l]
                        record_layer['top1_ratio']=(record_layer['top1_ratio']/record_layer['total_num']).round(4)
                        record_layer['scores']=(record_layer['scores']/record_layer['total_num']).round(4)
                        if verbose:
                            print(l, 'top1_ratio:', record_layer['top1_ratio'])                        
                        record_layer['unevenness'] = unevenness_measure(record_layer['top1_ratio'])
                    if verbose:
                        print('unevenness:', [round(record_layers[l]['unevenness'],3) for l in record_layers])
    return dataset2gate_record
def compare_unevenness(dataset2gate_record, common_attributes, compared_attributes):
    #common_attributes = {'dataset':dataset, 'stack_mode':stack_mode, 'attention_mode':attention_mode, 'module':module, 'layer_index':layer_index}
    #compare_attributes = {'dataset':dataset, 'stack_mode':stack_mode, 'attention_mode':attention_mode, 'module':module, 'layer_index':layer_index}
    compare_count = {}
    for attribute in compared_attributes:
        if compared_attributes[attribute] is not None:
            compare_count[attribute] = {value:{'sum':0, 'count':0} for value in compared_attributes[attribute]}
    for dataset in dataset2gate_record:
        if common_attributes['dataset'] is not None and dataset not in common_attributes['dataset']:
            continue
        gate_record = dataset2gate_record[dataset]
        for stack_mode in gate_record:
            if common_attributes['stack_mode'] is not None and stack_mode not in common_attributes['stack_mode']:
                continue
            for attention_mode in gate_record[stack_mode]:
                if common_attributes['attention_mode'] is not None and attention_mode not in common_attributes['attention_mode']:
                    continue
                for module in gate_record[stack_mode][attention_mode]:
                    if common_attributes['module'] is not None and module not in common_attributes['module']:
                        continue
                    record_layers = gate_record[stack_mode][attention_mode][module]
                    
                    for layer_index in range(len(record_layers)):
                        if common_attributes['layer_index'] is not None and layer_index not in common_attributes['layer_index']:
                            continue
                        record_layer = record_layers[layer_index]
                        unevenness = record_layer['unevenness']
                        current_attributes = {'dataset':dataset, 'stack_mode':stack_mode, 'attention_mode':attention_mode, 'module':module, 'layer_index':layer_index}
                        for attribute in compare_count:
                            current_value = current_attributes[attribute]
                            if current_value in compare_count[attribute]:
                                compare_count[attribute][current_value]['sum'] += unevenness
                                compare_count[attribute][current_value]['count'] += 1
    for attribute in compare_count:
        print('attribute:', attribute)
        for value in compare_count[attribute]:
            average_unevenness = compare_count[attribute][value]['sum']/(compare_count[attribute][value]['count']+1e-4)
            compare_count[attribute][value]['average_unevenness'] = average_unevenness
            print('value:', value, 'average unevenness:', round(average_unevenness,4))
        print('common average', np.mean([compare_count[attribute][value]['average_unevenness'] for value in compare_count[attribute]]))
                        
if __name__ == '__main__':
    dataset2gate_record = process_gate_loads('/home/zkhu143/iuie/output_ssd2/re_moelora/all/no_cluster_embedding_for_gate/InstructUIE_addname_16_4_1_router_z_True_False_False_False/gate_loads.jsonl')
    common_attributes = {'dataset':None, 'stack_mode':'encoder', 'attention_mode':['self_attention'], 'module':None, 'layer_index':None}
    compared_attributes = {'dataset':None, 'stack_mode':None, 'attention_mode':None, 'module':None, 'layer_index':range(24)}
    compare_unevenness(dataset2gate_record, common_attributes, compared_attributes)

                
