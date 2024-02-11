import json
import os
import random
def rename_ontology_RE(ontology_map, dataset_dir, converted_dir):
    for set_name in ['train', 'dev', 'test']:
        with open('{}/{}.json'.format(dataset_dir, set_name)) as fin:
            data = json.load(fin)
        for i in range(len(data)):
            for r in range(len(data[i]['relations'])):
                data[i]['relations'][r]['type'] = ontology_map[data[i]['relations'][r]['type']]
        with open('{}/{}.json'.format(converted_dir, set_name), 'w') as fout:
            json.dump(data, fout, indent=2)
    mapped_ontology = [ontology_map[k] for k in ontology_map]
    with open('{}/labels.json'.format(converted_dir), 'w') as fout:
        json.dump(mapped_ontology, fout, indent=2)
def generate_no_label_dataset(task,origin_name,origin_dir,target_dir):
    train_data = json.load(open('{}/train.json'.format(origin_dir)))
    new_train_data = []
    for i in range(len(train_data)):
        if task == 'RE':
            new_train_data.append({'sentence':train_data[i]['sentence'],'relations':[]})
        else:
            raise NotImplementedError
    with open('{}/{}_train_no_label.json'.format(target_dir,origin_name),'w') as fout:
        json.dump(new_train_data,fout,indent=2)
def sample_trainset(task,origin_name,origin_dir,target_dir,sample_num):
    #set seed
    random.seed(0)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    train_data = json.load(open('{}/train.json'.format(origin_dir)))
    #shuffle
    random.shuffle(train_data)
    new_train_data = train_data[:sample_num]
    with open('{}/train.json'.format(target_dir),'w') as fout:
        json.dump(new_train_data,fout,indent=2)
    test_data = json.load(open('{}/test.json'.format(origin_dir)))
    with open('{}/test.json'.format(target_dir),'w') as fout:
        json.dump(test_data,fout,indent=2)
    dev_data = json.load(open('{}/dev.json'.format(origin_dir)))
    with open('{}/dev.json'.format(target_dir),'w') as fout:
        json.dump(dev_data,fout,indent=2)
    labels = open('{}/labels.json'.format(origin_dir)).readlines()
    with open('{}/labels.json'.format(target_dir),'w') as fout:
        fout.writelines(labels)
if __name__ == '__main__':
    #ontology_map = {'adverse effect': 'adverse'}
    #dataset_dir = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/ADE_corpus'
    #converted_dir = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/ADE_corpus_renamed'
    #if not os.path.exists(converted_dir):
    #    os.mkdir(converted_dir)
    #rename_ontology_RE(ontology_map, dataset_dir, converted_dir)
    #origin_dir = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/semval-RE'
    #target_dir = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/finred'
    #generate_no_label_dataset('RE','semval-RE',origin_dir,target_dir)
    origin_dir = './data/ie_instruct/NER/MultiNERD'
    target_dir = './data/ie_instruct/NER/MultiNERD_sample_20000'
    sample_trainset('NER','MultiNERD',origin_dir,target_dir,20000)