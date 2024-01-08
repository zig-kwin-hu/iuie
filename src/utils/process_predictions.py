import json
file_name = '/home/zkhu143/iuie/output/ner_lora/plo_all/iuie-xxl/eval_eval_predictions.jsonl'
with open(file_name, 'r') as f:
    lines = f.readlines()
    js = [json.loads(line) for line in lines]
length_count = {'sentence': [], 'label': [], 'prediction': []}
for j in js:
    sentence = j['Instance']['sentence']
    label = j['Instance']['ground_truth']
    prediction = j['Prediction']
    length_count['sentence'].append(len(sentence))
    length_count['label'].append(len(label))
    length_count['prediction'].append(len(prediction))
#print percentiles
import numpy as np
print('sentence',np.percentile(length_count['sentence'], [0, 25, 50, 75, 95, 100]))
print('label',np.percentile(length_count['label'], [0, 25, 50, 75, 95, 100]))
print('prediction',np.percentile(length_count['prediction'], [0, 25, 50, 75, 95, 100]))
