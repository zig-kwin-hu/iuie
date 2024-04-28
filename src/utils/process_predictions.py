import json
def process_predictions(file_name):
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
# example of one line in the file
#{"Task": "NER", "Dataset": "Broad_Tweet_Corpus", "Instance": {"id": "4", "sentence": "Study of ancient faeces suggests that Neanderthals had more of a taste for fruit and vegetables : http://t.co/HWgxzX1tZ1", "label": " None", "instruction": "Please tell me all the entity words in the text that belong to a given category.Output format is \"type1: word1; type2: word2\". \n Option: location, person, organization \nText: {0} \nAnswer:", "ground_truth": " None", "answer_prefix": "Answer:", "unique_id": "NER_Broad_Tweet_Corpus_test_1288"}, "Prediction": "None"}

def count_none(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        js = [json.loads(line) for line in lines]
    count = 0
    for j in js:
        g = j['Instance']['ground_truth']
        p = j['Prediction']
            count += 1
    print(count)
