"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import json

import math
import time
from typing import Dict, List, Optional
import evaluate

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import json
import os

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def prefix_match_em_score(references, predictions):
    """
    Compute the prefix match between references and predictions
    :param references: a list of strings
    :param predictions: a list of strings
    :return: the prefix match score
    """
    prefix_match_score = 0
    for ref, pred in zip(references, predictions):
        ref = ref.strip()
        if ref.endswith("END"):
            ref = ref[:-3] # remove the END token
        pred = pred.strip()
        if pred.startswith(ref):
            prefix_match_score += 1
    return prefix_match_score / len(references)

class QuestionAnsweringLMTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, is_squad = False, do_recite=False,  **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        pipeline_name = "text-generation"
        self.do_recite = do_recite
        self.is_qa = is_squad
        if self.do_recite:
            max_new_tokens = 384
        elif self.is_qa:
            max_new_tokens = 128
        else:
            max_new_tokens= 256
        print("Max new tokens", max_new_tokens)
        self.pipe = pipeline(pipeline_name, model=self.model, tokenizer=self.tokenizer, device = "cuda", max_new_tokens = max_new_tokens, return_full_text = False,
                        do_sample=False,pad_token_id = self.tokenizer.pad_token_id)
        self.eval_examples = eval_examples
        self.save_times = 0
    def extract_answer_from_text(self, text, splitter):
        if not splitter in text or len(text.split(splitter))>2:
            print("Wrongly formatted text {}".format(text))
            return "", ""
        recite, answer = text.split(splitter)
        recite = recite.strip()
        answer = answer.strip()
        return recite, answer

    def extract_recitations_and_answers_from_texts(self, text_lst, splitter = "Answer:"):
        """
        Extract the recitations and answers from a list of texts
        :param splitter: the splitting token between recitation and the answer
        :param text_lst: a list of texts, each with "{recitation} {splitter} {answer}"
        :return: a list of recitations and answers
        """
        recites = []
        answers = []
        for text in text_lst:
            recite, answer = self.extract_answer_from_text(text, splitter=splitter)
            recites.append(recite)
            answers.append(answer)
        return recites, answers

    def evaluate(self, eval_dataset=None, ignore_keys: Optional[List[str]] = None,  metric_key_prefix="eval"):
        prior_metrics = super().evaluate(eval_dataset=eval_dataset,ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        predictions = []
        for out in tqdm(self.pipe(KeyDataset(self.eval_examples, "inputs"), batch_size=16), total=len(self.eval_examples), desc = 'running evaluation'):
            predictions.append(out)
        predicted_text = [x[0]['generated_text'] for x in predictions]

        if self.is_qa:

            # exact_match_metric = evaluate.load("exact_match")
            # references = [self.extract_answer_from_text(t) for t in self.eval_examples['targets']]
            # metrics = exact_match_metric.compute(predictions = predicted_text, references = references)
            # print("References", references[:10], "Predicted answers", predicted_text[:10])
            bleu = evaluate.load("sacrebleu")
            references = self.eval_examples['targets']
            pred_result = {"raw": list(zip(references, predicted_text))}
            metrics = {}
            if self.do_recite:
                prediction_recitations, prediction_answers = self.extract_recitations_and_answers_from_texts(
                    predicted_text)
                gt_recitations, gt_answers = self.extract_recitations_and_answers_from_texts(self.eval_examples['targets'])
                recite_bleu = bleu.compute(predictions=prediction_recitations, references=gt_recitations)
                answer_bleu = bleu.compute(predictions=prediction_answers, references=gt_answers)
                pred_result['recite'] = list(zip(gt_recitations, prediction_recitations))
                pred_result['qa'] = list(zip(gt_answers, prediction_answers))
                exact_match = evaluate.load("exact_match")
                metrics["recite_bleu"] = recite_bleu['score']
                metrics["qa_bleu"] = answer_bleu['score']
                metrics['recite_exact_match'] = \
                exact_match.compute(predictions=prediction_recitations, references=gt_recitations)['exact_match']
                metrics['qa_exact_match'] = exact_match.compute(predictions=prediction_answers, references=gt_answers)[
                    'exact_match']
            else:
                prediction_answers = predicted_text
            squad_metric = evaluate.load("squad")
            references = [{'id': str(i), 'answers': row['answers']} for i,row in enumerate(self.eval_examples)]
            predictions = [{'id': str(i), 'prediction_text': x} for i,x in enumerate(prediction_answers)]
            print("References", references[-5:], "Predicted answers", predicted_text[-5:])
            metrics.update(squad_metric.compute(predictions=predictions,references=references))

        else:
            bleu = evaluate.load("sacrebleu")
            references = self.eval_examples['targets']
            pred_result = {"raw": list(zip(references, predicted_text))}
            exact_match = evaluate.load("exact_match")
            print("References v.s. predicted", list(zip(references, predicted_text))[:5])
            if len(predicted_text[0])==0:
                metrics = {}
            else:
                bleu_results = bleu.compute(predictions = predicted_text, references = references)
                metrics = {"bleu": bleu_results["score"]}
                metrics.update(exact_match.compute(predictions=[x.strip() for x in predicted_text], references=[x.strip() for x in references]))
                metrics['prefix_exact_match'] = prefix_match_em_score(references, predicted_text)
                if self.do_recite:
                    prediction_recitations, prediction_answers = self.extract_recitations_and_answers_from_texts(predicted_text)
                    gt_recitations, gt_answers = self.extract_recitations_and_answers_from_texts(references)
                    recite_bleu = bleu.compute(predictions = prediction_recitations, references = gt_recitations)
                    answer_bleu = bleu.compute(predictions = prediction_answers, references = gt_answers)
                    pred_result['recite'] = list(zip(gt_recitations, prediction_recitations))
                    pred_result['qa'] = list(zip(gt_answers, prediction_answers))
                    metrics["recite_bleu"] = recite_bleu['score']
                    metrics["qa_bleu"] =  answer_bleu['score']
                    metrics['recite_exact_match'] = exact_match.compute(predictions = prediction_recitations, references = gt_recitations)['exact_match']
                    metrics['qa_exact_match'] = exact_match.compute(predictions = prediction_answers, references = gt_answers)['exact_match']
        # save the generation outcome to disk
        SAVE_PRED_DIR = os.getenv("SAVE_PRED_DIR")
        if SAVE_PRED_DIR is not None:
            suffix = self.save_times
            filename = os.path.join(SAVE_PRED_DIR, f"predictions_{metric_key_prefix}_{suffix}.json")
            print("Saved prediction result to", filename)
            self.save_times += 1
            json.dump(pred_result, open(filename, "w"))
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        print(metrics)
        if self.args.should_log:
            # Only the main node log the results by default
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        prior_metrics.update(metrics)
        return prior_metrics


    def predict(
            self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test", **gen_kwargs
    ):
        self._gen_kwargs = gen_kwargs.copy()

        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

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
        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)