import logging

import torch
from transformers.data.data_collator import *

import IPython
logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5', 'ZWK/InstructUIE', './huggingface/InstructUIE']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None
    return_loss_mask: bool = True
    uid2sentid: dict = None
    uid2clusterid: dict = None
    sentence_embeddings_for_gate: Optional[Any] = None
    cluster_embeddings_for_gate: Optional[Any] = None
    write_gate_loads: bool = False


    #huzikun log
    #input_ids: list = None
    #source_lens: list = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            raise ValueError('Unsupport model {}!'.format(model_name))

        return model_inputs

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        # TODO, support few shot
        # add few shot samples
        samples = ''
        if len(instance['Samples']) > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content
        # TODO, fix bug
        try:
            instruction = instruction.format(content)
        finally:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []
        sentids = []
        clusterids = []
        unique_ids = []

        for instance in batch:
            unique_id = instance['Instance']['unique_id']
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)
            sentid = self.uid2sentid[unique_id] if self.uid2sentid is not None else None
            clusterid = self.uid2clusterid[unique_id] if self.uid2clusterid is not None else None
            sentids.append(sentid)
            clusterids.append(clusterid)
            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            unique_ids.append(unique_id)
        sentence_embedding_for_gate, cluster_embedding_for_gate = None, None
        if self.sentence_embeddings_for_gate is not None and self.uid2sentid is not None:
            sentence_embedding_for_gate = self.sentence_embeddings_for_gate[sentids]
        if self.cluster_embeddings_for_gate is not None and self.uid2clusterid is not None:
            cluster_embedding_for_gate = self.cluster_embeddings_for_gate[clusterids]
        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_token_type_ids=False
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
            #huzikun log
            #if self.input_ids is None:
            #    self.input_ids = []
            #if self.source_lens is None:
            #    self.source_lens = []
            #self.input_ids.append(model_inputs['input_ids'].shape[1])
            #self.source_lens.extend([len(s) for s in sources])
            #fout = open('temp.txt','a')
            #for s in sources:
            #    if len(s) >= 1808:
            #        fout.write(s+'\n')
            #if len(self.input_ids) >= 532:
            #    #sort and output percentiles of input_ids
            #    self.input_ids.sort()
            #    self.source_lens.sort()
            #    import numpy as np
            #    percentiles = np.percentile(self.input_ids, range(0, 101, 1))
            #    print('input_ids percentiles',percentiles)
            #    percentiles = np.percentile(self.source_lens, range(0, 101, 1))
            #    print('source_lens percentiles',percentiles)


            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs['sentence_embedding_for_gate'] = sentence_embedding_for_gate
        model_inputs['cluster_embedding_for_gate'] = cluster_embedding_for_gate
        if self.write_gate_loads:
            model_inputs['unique_ids'] = unique_ids

            #self._save_samples(model_inputs, sources, labels)
        '''
        model_inputs['task'] = []
        model_inputs['dataset'] = []
        for i, instance in enumerate(batch):
            model_inputs['task'].append(instance['Instance']['task'])
            model_inputs['dataset'].append(instance['Instance']['dataset'])
        '''
        
        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1
        if batch[0]['subset'] == "train":
            limit_input_len = self.max_source_length + self.max_target_length
        else:
            limit_input_len = self.max_source_length

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            task_input = instruction
            label = label + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(task_input)["input_ids"]
            tokenized_label = self.tokenizer(label)["input_ids"]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['dev', 'test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    input_wo_label = self.tokenizer.decode(
                        tokenized_input[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
            else:
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = self.max_source_length
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, 'labels': labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_token_type_ids=False
            )

            label_mask = model_inputs["attention_mask"].bool()
            model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask, self.label_pad_token_id)

            if self.return_loss_mask:
                # loss mask
                max_len = min(max_len, limit_input_len)
                loss_mask = torch.ones((label_mask.shape))
                for k, label_len in enumerate(label_lens):
                    loss_mask[k, : max_len - label_len - 1] = 0
                model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        loss_label = []
        # if hasattr(model_inputs, 'loss_mask'):
        #     for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
        #         loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

        #     with open(self.input_record_file, 'w', encoding='utf-8') as f:
        #         for text, label, mask_label in zip(sources, labels, loss_label):
        #             f.write(text+'\n')
        #             f.write(label + '\n')
        #             f.write(mask_label+'\n\n')
        # else:
        #     with open(self.input_record_file, 'w', encoding='utf-8') as f:
        #         need_decode = False
        #         if hasattr(labels, 'input_ids'):
        #             labels = labels['input_ids']
        #             need_decode = True
                
        #         for text, label in zip(sources, labels):
        #             f.write(text + '\n')
        #             if need_decode:
        #                 label = self.tokenizer.decode(label, clean_up_tokenization_spaces=False)
        #             f.write(label + '\n')