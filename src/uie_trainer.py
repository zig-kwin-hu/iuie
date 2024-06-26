import copy
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_zero3_enabled
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_safetensors_available,
    logging,
)

import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers import EarlyStoppingCallback

from transformers.trainer_pt_utils import nested_truncate, nested_concat, nested_numpify
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from peft_moe import get_peft_model_state_dict

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset import ANSWER_PREFIX
import os
import json
import numpy as np
import IPython
from typing import NamedTuple, Optional, Union, Tuple, Dict, List, Callable, Iterable
from transformers.utils import logging
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
if is_peft_available():
    from peft import PeftModel
    from peft_moe import PeftModel as PeftMoeModel
if is_safetensors_available():
    import safetensors.torch


logger = logging.get_logger(__name__)

class UIEPredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    embeddings: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]]

def skip_instructions(model, predictions_ids, tokenizer, dataset, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for example, pred in zip(dataset, predictions):
            if example['Instance']['answer_prefix'] in pred:
                splits = pred.split(example['Instance']['answer_prefix'])
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions

class CallbackHandlerWithTrainer(CallbackHandler):
    """Internal class that just calls the list of callbacks in order."""

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, trainer, trial, non_saving_datasets):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, trainer=trainer, trial=trial, non_saving_datasets=non_saving_datasets)

class SkipEpochEvalCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Evaluate and save checkpoint only every `args.skip_epoch_eval` epochs
        assert (args.evaluation_strategy in [IntervalStrategy.EPOCH, IntervalStrategy.STEPS, IntervalStrategy.NO])
        assert (args.save_strategy in [IntervalStrategy.EPOCH, IntervalStrategy.STEPS, IntervalStrategy.NO])
        if args.evaluation_strategy == IntervalStrategy.EPOCH and self.save_strategy == IntervalStrategy.EPOCH and\
            args.skip_epoch_eval > 0:
            if int(state.epoch) % args.skip_epoch_eval != 0:
                control.should_evaluate = False
                control.should_save = False

        return control

class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control

class SaveMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, model = None, trainer = None, trial = None, non_saving_datasets=[], **kwargs):
        # Save metrics after each evaluation
        epoch = state.epoch
        
        if args.local_rank == 0:
            if args.use_test_as_eval:
                output_path = os.path.join(args.output_dir, f"eval_metrics_each_epoch_use_test_as_eval_{args.max_num_instances_per_predict_task}.jsonl")
                #output_path = os.path.join(args.output_dir, f"eval_metrics_each_epoch_use_test_as_eval.jsonl")
            else:
                output_path = os.path.join(args.output_dir, f"eval_metrics_each_epoch_{args.max_num_instances_per_predict_task}.jsonl")
                #output_path = os.path.join(args.output_dir, f"eval_metrics_each_epoch.jsonl")
            if epoch <= 1 and args.evaluation_strategy == IntervalStrategy.EPOCH:
                f = open(output_path, "w")
            else:
                f = open(output_path, "a+")
            f.write(json.dumps({'eval':metrics})+'\n')
            f.close()

        return control

class SavePeftModelCallback(TrainerCallback):
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        return control

class SaveBestModelsCallback(TrainerCallback):
    # created by zikun, save the best model for each dataset after each evaluation. So as a result, there will be multiple best models saved.
    # during training, there will be a dict storing the best_f1_for_${dataset_name} for each dataset
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.best_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, model = None, trainer = None, trial = None, non_saving_datasets=[], **kwargs):
        assert isinstance(trainer, UIETrainer)
        assert model is not None
        for metric_name, metric_value in metrics.items():
            if 'eval_F1_for_' not in metric_name:
                continue
            dataset_name = metric_name.split('eval_F1_for_')[1]
            task_name = None
            if '|' in dataset_name:
                dataset_name, task_name = dataset_name.split('|')
                assert task_name in ['EE', 'EET', 'EEA', 'RE', 'NER']
            if dataset_name in non_saving_datasets:
                continue
            folder_name = f"best_model_for_{dataset_name}|{task_name}" if task_name is not None else f"best_model_for_{dataset_name}"
            checkpoint_folder = os.path.join(
                    args.output_dir, folder_name
            )
            # Modified: use self.best_metrics to track result due to read/write file in multi processing would cause file sync error 
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = 0
            if args.local_rank == 0 and not os.path.exists(checkpoint_folder):
                os.mkdir(checkpoint_folder)
                with open(os.path.join(checkpoint_folder, f'best_metrics_{args.max_num_instances_per_predict_task}.json'), 'w') as f:
                    f.write(json.dumps({metric_name:0})+'\n')
            
            if metric_value > self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = metric_value
                if args.local_rank == 0:
                    with open(os.path.join(checkpoint_folder, f'best_metrics_{args.max_num_instances_per_predict_task}.json'), 'w') as f:
                        f.write(json.dumps({metric_name:metric_value})+'\n')
                        f.write(json.dumps(metrics)+'\n')
                if not args.no_saving:
                    trainer._save_checkpoint(model, trial, metrics=None, checkpoint_folder=folder_name)
                print(f"best model for {dataset_name}|{task_name} saved at {checkpoint_folder}")
        return control
class EarlyStoppingCallbackWithLog(EarlyStoppingCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or \
        (operator(metric_value, state.best_metric) 
         and abs(metric_value - state.best_metric) > self.early_stopping_threshold):
            self.early_stopping_patience_counter = 0
            with open(os.path.join(args.output_dir, 'early_stopping_log.jsonl'), 'a+') as f:
                f.write('set best metric, '+'state.best_metric: '+str(state.best_metric)+' metric_value: '+str(metric_value)+'\n')
        else:
            self.early_stopping_patience_counter += 1
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
        with open(os.path.join(args.output_dir, 'early_stopping_log.jsonl'), 'a+') as f:
           f.write(json.dumps({'global_step':state.global_step, 'metric_value':metric_value, \
                               'early_stopping_patience_counter':self.early_stopping_patience_counter, \
                                   'should_training_stop':str(control.should_training_stop)})+'\n')
class UIETrainer(Seq2SeqTrainer):
    #modified by huzikun, use customized callback handler
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        non_saving_datasets = [],
        max_new_tokens=50,
        num_beams=1,
        repetition_penalty=1.0,
        pad_token_id=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        
        # Rewrite: Override callback handler
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandlerWithTrainer(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        
        self.non_saving_datasets = non_saving_datasets

        #init params for predictions
        self.predict_dataset = predict_dataset
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.pad_token_id = pad_token_id
        if self.args.write_gate_loads:
            f = open(os.path.join(self.args.output_dir, 'gate_loads.jsonl'),'w')
            f.close()
        

    # Rewrite the evaluation function, with customized call of evaluate, passing the trainer and trial object so that we can modify the save process.
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                        pad_token_id=self.pad_token_id,
                    )
                    metrics.update(dataset_metrics)
            else:
                # Rewrite: Modified evaluate by passing additional trial param
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, trial=trial)
            
            if self.args.test_with_eval:
                predict_results = self.predict(
                    self.predict_dataset,
                    metric_key_prefix="predict",
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.pad_token_id
                )
                metrics.update(predict_results.metrics)

                if self.args.local_rank == 0:
                    predict_metrics = predict_results.metrics
                    output_path = os.path.join(self.args.output_dir, f"test_metrics_each_epoch_{self.args.max_num_instances_per_predict_task}.jsonl")
                    predict_metrics['epoch'] = epoch
                    if epoch <= 1 and self.args.evaluation_strategy == IntervalStrategy.EPOCH:
                        f = open(output_path, "w")
                    else:
                        f = open(output_path, "a+")
                    self.log_metrics("predict", predict_metrics)
                    f.write(json.dumps({'test':predict_metrics})+'\n')
                    f.close()
                
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            if not self.args.no_saving:
                self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # modified by huzikun, rewrite the evaluate function, with customized call of on_evaluate, passing the trainer and trial object so that we can modify the save process.
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial = None,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        # Rewrite: passing additional trial and non_saving_datasets params
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics, trainer=self, trial=trial, non_saving_datasets=self.non_saving_datasets)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.model_wrapped is self.model:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        embeddings_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_embeddings = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            cluster_embedding_for_gate = inputs.get("cluster_embedding_for_gate", None)
            assert not (self.args.use_cluster_embedding_for_gate and cluster_embedding_for_gate is None)
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            embeddings = None
            # Prediction step
            if self.args.embedding_type is not None:
                loss, logits, labels, embeddings = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            else:
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                #labels = self._pad_across_processes(labels)
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                #inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                #logits = self._pad_across_processes(logits)
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if embeddings is not None:
                embeddings = self._nested_gather(embeddings)
                embeddings_host = embeddings if embeddings_host is None else torch.cat((embeddings_host, embeddings), dim=0)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if embeddings_host is not None:
                    embeddings = nested_numpify(embeddings_host)
                    all_embeddings = (embeddings if all_embeddings is None else np.concatenate((all_embeddings, embeddings), axis=0))
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host, embeddings_host = None, None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if embeddings_host is not None:
            embeddings = nested_numpify(embeddings_host)
            all_embeddings = (embeddings if all_embeddings is None else np.concatenate((all_embeddings, embeddings), axis=0))
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        if all_embeddings is not None:
            all_embeddings = all_embeddings[:num_samples]

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        if self.args.embedding_type is not None:
            return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples), all_embeddings, all_inputs
        else:
            return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if (not self.args.predict_with_generate or prediction_loss_only) and (not self.args.embedding_type):
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        if self.args.embedding_type is not None:
            inputs["output_hidden_states"] = True
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        #default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else False#default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        
        if not gen_kwargs["num_beams"]:
            gen_kwargs["num_beams"] = 1
        if not gen_kwargs.get("max_new_tokens"):
            gen_kwargs["max_new_tokens"] = gen_kwargs["max_length"]
        #if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            #gen_kwargs.pop("num_beams")
        #if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            #gen_kwargs.pop("max_length")  
        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        #generation_inputs = inputs.copy()

        #if (
        #    "labels" in generation_inputs
        #    and "decoder_input_ids" in generation_inputs
        #    and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        #):
        #    generation_inputs = {
        #        k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
        #    }
        #print('gen_kwargs',gen_kwargs)
        #"/home/zkhu142/anaconda3/envs/llama/lib/python3.8/site-packages/transformers/generation/utils.py", line 1297, 
        sentence_embedding_for_gate = inputs.get("sentence_embedding_for_gate", None)
        cluster_embedding_for_gate = inputs.get("cluster_embedding_for_gate", None)
        assert not (self.args.use_sentence_embedding_for_gate and self.args.use_cluster_embedding_for_gate)
        embedding_for_gate = None
        if sentence_embedding_for_gate is not None and self.args.use_sentence_embedding_for_gate:
            embedding_for_gate = sentence_embedding_for_gate
        elif cluster_embedding_for_gate is not None and self.args.use_cluster_embedding_for_gate:
            embedding_for_gate = cluster_embedding_for_gate
        before_moe_lora_gate_embedding_reduction = getattr(self.model, "before_moe_lora_gate_embedding_reduction", None)
        if before_moe_lora_gate_embedding_reduction is not None and before_moe_lora_gate_embedding_reduction > 0 and embedding_for_gate is not None:
            embedding_for_gate = model.moe_lora_gate_embedding_reduction.to(embedding_for_gate.dtype)(embedding_for_gate)
            embedding_for_gate = model.moe_lora_gate_embedding_reduction_dropout(embedding_for_gate)
        generated_tokens = self.model.generate(
            #**generation_inputs,
            #**gen_kwargs,
            input_ids=generation_inputs,
            generation_config=generation_config,
            embedding_for_gate=embedding_for_gate,
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    '''
                    #need to be deleted
                    tasks = inputs.pop('task')
                    datasets = inputs.pop('dataset')
                    '''
                    sentence_embedding_for_gate = inputs.pop("sentence_embedding_for_gate", None)
                    cluster_embedding_for_gate = inputs.pop("cluster_embedding_for_gate", None)
                    assert not (self.args.use_sentence_embedding_for_gate and self.args.use_cluster_embedding_for_gate)
                    assert not (self.args.use_cluster_embedding_for_gate and cluster_embedding_for_gate is None)
                    if sentence_embedding_for_gate is not None and self.args.use_sentence_embedding_for_gate:
                        inputs['embedding_for_gate'] = sentence_embedding_for_gate
                    elif cluster_embedding_for_gate is not None and self.args.use_cluster_embedding_for_gate:
                        inputs['embedding_for_gate'] = cluster_embedding_for_gate
                    unique_ids = inputs.pop('unique_ids',None)

                    before_moe_lora_gate_embedding_reduction = getattr(model, 'before_moe_lora_gate_embedding_reduction', None)
                    if before_moe_lora_gate_embedding_reduction is None:
                        before_moe_lora_gate_embedding_reduction = getattr(model.base_model, 'moe_lora_gate_embedding_reduction', None)
                    if before_moe_lora_gate_embedding_reduction is not None and before_moe_lora_gate_embedding_reduction>0 and\
                        'embedding_for_gate' in inputs and inputs['embedding_for_gate'] is not None:
                        #model.moe_lora_gate_embedding_reduction.weight.dtype bfloat16, inputs['embedding_for_gate'].dtype float32
            
                        inputs['embedding_for_gate'] = model.moe_lora_gate_embedding_reduction.to(inputs['embedding_for_gate'].dtype)(inputs['embedding_for_gate'])
                        inputs['embedding_for_gate'] = model.moe_lora_gate_embedding_reduction_dropout(inputs['embedding_for_gate'])
                    outputs = model(**inputs)
                    
                    if self.args.embedding_type is not None:
                        encoder_last_h = outputs["encoder_last_hidden_state"]
                        decoder_last_h = outputs["decoder_hidden_states"][-1]
                        embeddings = self._get_embedding(encoder_last_h, decoder_last_h)
                    if self.args.write_gate_loads:
                        
                        fout = open(os.path.join(self.args.output_dir, 'gate_loads.jsonl'),'a+')
                        for mode in ['self_attention','ff']:
                            encoder_gate_loads = self.record_gate_loads(outputs.encoder_gate_outputs[mode], lora_part=mode,
                                 stack_mode='encoder', unique_ids=unique_ids)
                            for encoder_gate_load in encoder_gate_loads:
                                fout.write(json.dumps(encoder_gate_load)+'\n')
                        for mode in ['self_attention','cross_attention','ff']:
                            decoder_gate_loads = self.record_gate_loads(outputs.decoder_gate_outputs[mode], lora_part=mode,
                             stack_mode='decoder', unique_ids=unique_ids)
                            for decoder_gate_load in decoder_gate_loads:
                                fout.write(json.dumps(decoder_gate_load)+'\n')
                        fout.close()
                    '''
                    #need to be deleted
                    decoded_strings = self.tokenizer.batch_decode(inputs['input_ids'])
                    embeddings_np = embeddings.cpu().numpy()
                    tosave = []
                    for i, decoded_string in enumerate(decoded_strings):
                        tosave.append({'decoded':decoded_string, 'embedding':embeddings_np[i][:10].tolist(), 'task':tasks[i], 'dataset':datasets[i]})
                    output_path = os.path.join(self.args.output_dir, f"test_embeddings.jsonl")
                    if not os.path.exists(output_path):
                        f = open(output_path, "w")
                    else:
                        f = open(output_path, "a+")
                    for item in tosave:
                        f.write(json.dumps(item)+'\n')
                    f.close()
                    '''
                                    
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None
        if self.args.embedding_type is not None:
            return (loss, generated_tokens, labels, embeddings)
        return (loss, generated_tokens, labels)
    def _get_embedding(self, encoder_last_h, decoder_last_h):
        if self.args.embedding_type == 'mean_of_encoder':
            embedding = encoder_last_h.mean(dim=1)
        elif self.args.embedding_type == 'first_of_decoder':
            embedding = decoder_last_h[:, 0, :]
        elif self.args.embedding_type == 'last_of_decoder':
            #considering the padding token
            raise NotImplementedError('last_of_decoder is not implemented yet')
        else:
            raise NotImplementedError('embedding_type {} is not implemented yet, please choose from [mean_of_encoder, first_of_decoder, last_of_decoder]'.format(self.args.embedding_type))
        return embedding
    # modified by huzikun, rewrite the _save_checkpoint function, with customizable checkpoint folder name
    def _save_checkpoint(self, model, trial, metrics=None, checkpoint_folder=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Modified: added if check to use given folder name or default name
        # Save model checkpoint
        if checkpoint_folder is None:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

                
        if self.args.use_lora and self.args.save_lora_weights_only:
            # Modified: if use_lora is True, model will be PeftModel and the save_pretrained only save adapter_model.bin and adapter_config.json
            self.model.save_pretrained(output_dir)
        else:
            # Transformer trainer._save_checkpoint original code
            self.save_model(output_dir, _internal_call=True)
            if self.is_deepspeed_enabled:
                # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
                # config `stage3_gather_16bit_weights_on_model_save` is True
                self.model_wrapped.save_checkpoint(output_dir)

            # Save optimizer and scheduler
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer.consolidate_state_dict()

            if self.fsdp:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
            elif is_sagemaker_mp_enabled():
                opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
                smp.barrier()
                if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                    smp.save(
                        opt_state_dict,
                        os.path.join(output_dir, OPTIMIZER_NAME),
                        partial=True,
                        v3=smp.state.cfg.shard_optimizer_state,
                    )
                if self.args.should_save:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
                    if self.do_grad_scaling:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
            elif self.args.should_save and not self.is_deepspeed_enabled:
                # deepspeed.save_checkpoint above saves model/optim/sched
                if self.fsdp:
                    torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
                else:
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
            
            
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
        # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()
        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()
        
        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        
        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)
        
        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        # Modified: Save lora weights on deepspeed zero3. This part must process after original code to override the empty adapter_model.bin due to incompatible between zero3 and peft
        if self.args.use_lora and self.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
            # save adapter config
            # self.model.peft_config.save_pretrained(output_dir)
            # get state dict through deepspeed engine
            engine_state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            lora_state_dict = get_peft_model_state_dict(self.model, engine_state_dict)
            
            if self.args.local_rank == 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Save adapter model at {output_dir}")
    #modified by huzikun, make it able to return the embeddings
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs
        
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        if self.args.embedding_type is not None:
            output, all_embeddings, all_inputs = eval_loop(
                test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )
        else:
            output = eval_loop(
                test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        if self.args.embedding_type is not None:
            return UIEPredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics, embeddings=all_embeddings, inputs=all_inputs)
        else:
            return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    #modified by huzikun, add condition to check whether the model is PeftModel or PeftMoeModel, if so, load adapter model
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, self.state.best_model_checkpoint)
            else:
                has_been_loaded = True
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                        # If the 'user_content.pt' file exists, load with the new smp api.
                        # Checkpoint must have been saved with the new smp api.
                        smp.resume_from_checkpoint(
                            path=self.state.best_model_checkpoint,
                            tag=WEIGHTS_NAME,
                            partial=False,
                            load_optimizer=False,
                        )
                    else:
                        # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                        # Checkpoint must have been saved with the old smp api.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        state_dict["_smp_is_partial"] = False
                        load_result = model.load_state_dict(state_dict, strict=True)
                elif self.is_fsdp_enabled:
                    self.accelerator.state.fsdp_plugin.load_model(
                        self.accelerator, model, self.state.best_model_checkpoint
                    )
                else:
                    #modified by huzikun, add condition to load adapter model
                    if is_peft_available() and (isinstance(model, PeftModel) or isinstance(model, PeftMoeModel)):
                        # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                            if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                                model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                                # Load_adapter has no return value present, modify it when appropriate.
                                from torch.nn.modules.module import _IncompatibleKeys

                                load_result = _IncompatibleKeys([], [])
                            else:
                                logger.warning(
                                    "The intermediate checkpoints of PEFT may not be saved correctly, "
                                    f"using `TrainerCallback` to save {ADAPTER_WEIGHTS_NAME} in corresponding folders, "
                                    "here are some examples https://github.com/huggingface/peft/issues/96"
                                )
                                has_been_loaded = False
                        else:
                            logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                            has_been_loaded = False
                    else:
                        # We load the model state dict on the CPU to avoid an OOM error.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        # If the model is on the GPU, it still works!
                        # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                        # which takes *args instead of **kwargs
                        load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        sentence_embedding_for_gate = inputs.pop("sentence_embedding_for_gate", None)
        cluster_embedding_for_gate = inputs.pop("cluster_embedding_for_gate", None)
        assert not (self.args.use_sentence_embedding_for_gate and self.args.use_cluster_embedding_for_gate)
        if sentence_embedding_for_gate is not None and self.args.use_sentence_embedding_for_gate:
            inputs['embedding_for_gate'] = sentence_embedding_for_gate
        elif cluster_embedding_for_gate is not None and self.args.use_cluster_embedding_for_gate:
            inputs['embedding_for_gate'] = cluster_embedding_for_gate
        unique_ids = inputs.pop('unique_ids',None)
        
        before_moe_lora_gate_embedding_reduction = getattr(model, 'before_moe_lora_gate_embedding_reduction', None)        
        if before_moe_lora_gate_embedding_reduction is not None and before_moe_lora_gate_embedding_reduction>0 and\
              'embedding_for_gate' in inputs and inputs['embedding_for_gate'] is not None:
            #model.moe_lora_gate_embedding_reduction.weight.dtype bfloat16, inputs['embedding_for_gate'].dtype float32
            
            inputs['embedding_for_gate'] = model.moe_lora_gate_embedding_reduction.to(inputs['embedding_for_gate'].dtype)(inputs['embedding_for_gate'])
            inputs['embedding_for_gate'] = model.moe_lora_gate_embedding_reduction_dropout(inputs['embedding_for_gate'])

        outputs = model(**inputs)
        if self.args.write_gate_loads:
            fout = open(os.path.join(self.args.output_dir, 'gate_loads.jsonl'),'a+')
            for mode in ['self_attention','ff']:
                encoder_gate_loads = self.record_gate_loads(outputs.encoder_gate_outputs[mode], lora_part=mode,
                    stack_mode='encoder', unique_ids=unique_ids)
                for encoder_gate_load in encoder_gate_loads:
                    fout.write(json.dumps(encoder_gate_load)+'\n')
            for mode in ['self_attention','cross_attention', 'ff']:
                decoder_gate_loads = self.record_gate_loads(outputs.decoder_gate_outputs[mode], lora_part=mode,
                 stack_mode='decoder', unique_ids=unique_ids)
                for decoder_gate_load in decoder_gate_loads:
                    fout.write(json.dumps(decoder_gate_load)+'\n')
            fout.close()
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if 'gate_loss' in outputs and outputs['gate_loss'] is not None:
            loss = loss + outputs['gate_loss']*self.args.gate_loss_weight
        return (loss, outputs) if return_outputs else loss

    def record_gate_loads(self, gate_outputs, lora_part, stack_mode, unique_ids, prefix=''):
        records = []
        def recursive_round(element, decimal):
            if isinstance(element, list):
                return [recursive_round(sub_element, decimal) for sub_element in element]
            else:
                return round(element, decimal)
        for layer_index, gate_output in enumerate(gate_outputs):
            #Futural update: given a model, you could not know the target lora/moe_lora module names. 
            #So you need to dynamically select possible modules based on the model_name which can be retrieved from the model's config.
            gate_load = {}
            for module in ['q','k','v', 'wi_0','wi_1','wi','wo']:
                if (module not in gate_output) or (gate_output[module] is None) or ('top1_ratio' not in gate_output[module]):
                    gate_load[module] = None
                    continue
                gate_load[module] = {}

                gate_load[module]['top1_ratio'] = recursive_round(gate_output[module]['top1_ratio'].cpu().numpy().tolist(),4)
                gate_load[module]['scores'] = recursive_round(gate_output[module]['scores'].cpu().numpy().tolist(),4)
                assert len(gate_load[module]['top1_ratio']) == len(unique_ids)
            for index, unique_id in enumerate(unique_ids):
                gate_load_i = copy.deepcopy(gate_load)
                for module in ['q','k','v', 'wi_0','wi_1','wi','wo']:
                    if module in gate_load_i and gate_load_i[module] is not None:
                        gate_load_i[module]['top1_ratio'] = gate_load_i[module]['top1_ratio'][index]
                        gate_load_i[module]['scores'] = gate_load_i[module]['scores'][index]
                record = {'lora_part':lora_part, 'stack_mode':stack_mode, 'unique_id':unique_id, 'gate_load':gate_load_i, 'layer_index':layer_index}
                records.append(record)
        return records
        