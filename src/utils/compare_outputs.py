import json
import os
import copy
def compare_predictions_ner(predictions1, predictions2, target_path, name1, name2):
    fout = open(target_path, 'w')
    differences = []
    for prediction1, prediction2 in zip(predictions1, predictions2):
        assert prediction1['Instance']['unique_id'] == prediction2['Instance']['unique_id']
        if prediction1['Prediction'] != prediction2['Prediction']:
            different = copy.deepcopy(prediction1)
            different.pop('Prediction')
            different['Prediction '+ name1] = prediction1['Prediction']
            different['Prediction '+ name2] = prediction2['Prediction']
            differences.append(different)
    fout.write(json.dumps(differences, indent=4))
    fout.close()
    return differences
if __name__ == '__main__':
    predictions1 = [json.loads(l) for l in open('/home/zkhu143/iuie/output_ssd2/ner_lora/Broad_Tweet_Corpus_WikiANN_en/InstructUIE_16/test_eval_predictions_broad.jsonl').readlines()]
    predictions2 = [json.loads(l) for l in open('/home/zkhu143/iuie/output_ssd2/ner_lora/Broad_Tweet_Corpus_WikiANN_en/InstructUIE_16/test_eval_predictions_wikiann.jsonl').readlines()]
    predictions1 = [p for p in predictions1 if p['Dataset'] in ['Broad_Tweet_Corpus', 'WikiANN_en']]
    predictions2 = [p for p in predictions2 if p['Dataset'] in ['Broad_Tweet_Corpus', 'WikiANN_en']]
    differences = compare_predictions_ner(predictions1, predictions2, '/home/zkhu142/iuie/output_ssd2/ner_lora/Broad_Tweet_Corpus_WikiANN_en/InstructUIE_16/differences.json',
                                          'broad', 'wikiann')
    print(len(predictions1))
    print(len(predictions2))
    print(len(differences))
            