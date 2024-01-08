import json
import os
def generate_config_files_per_task(task, datasets, config_dir):
    name = '_'.join(datasets)
    output_dir = os.path.join(config_dir, '{}_configs'.format(task.lower()), name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subset in ['test_tasks.json','dev_tasks.json','train_tasks.json']:
        output_path = os.path.join(output_dir, subset)
        config = {task:[]}
        for dataset in datasets:
            config[task].append({
	        "sampling strategy": "full",
		    "dataset name": dataset
		})
        json.dump(config, open(output_path, 'w'), indent=4)
if __name__ == '__main__':
    config_dir = './configs'
    generate_config_files_per_task('NER', ['MultiNERD_sample_20000', 'bc5cdr', 'ncbi'], config_dir)
    generate_config_files_per_task('NER', ['MultiNERD_sample_20000'], config_dir)
    generate_config_files_per_task('NER', ['bc5cdr'], config_dir)    
    generate_config_files_per_task('NER', ['ncbi'], config_dir)
    generate_config_files_per_task('NER', ['finred'], config_dir)
    generate_config_files_per_task('NER', ['PolyglotNER_sample_20000', 'Broad_Tweet_Corpus', 'WikiANN_en', 'WikiNeural_sample_20000'], config_dir)
    generate_config_files_per_task('NER', ['PolyglotNER_sample_20000'], config_dir)
    generate_config_files_per_task('NER', ['Broad_Tweet_Corpus'], config_dir)
    generate_config_files_per_task('NER', ['WikiANN_en'], config_dir)
    generate_config_files_per_task('NER', ['WikiNeural_sample_20000'], config_dir)