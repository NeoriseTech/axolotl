'''Calculate metrics during evaluation.'''

import logging

import evaluate
import numpy as np
from transformers import PreTrainedTokenizer

LOG = logging.getLogger("axolotl.utils.metrics")

def get_metric(metric_name: str, tokenizer: PreTrainedTokenizer):
    if metric_name is None:
        return None
    if metric_name == 'f1':
        return F1Metric(tokenizer)
    elif metric_name == 'rouge':
        return RougeMetric(tokenizer)
    elif metric_name == 'bleu':
        return BleuMetric(tokenizer)
    else:
        ValueError(f"Metric {metric_name} not supported")

class Metric:
    def __init__(self, metric_name, compute_metrics, optimize_higher):
        self.metric_name = metric_name
        self.compute_metrics = compute_metrics
        self.optimize_higher = optimize_higher
        self.preprocess_logits = True

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        # Select the token ID with the highest prediction probability from the model's output at each sequence position
        return logits.argmax(dim=-1)
    
    def _strip_padding(self, logits, labels):
        # predictions have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels because the model is predicting
        # the next token in the sequence

        # Initialize lists to hold processed labels and predictions
        processed_labels = []
        processed_predictions = []

        # Loop over each sample in the batch. We want to strip off the padding tokens
        # because we only want to calculate metrics on the non-padded tokens. We must ensure
        # the matching labels and predictions are the same length so we may need to truncate
        # whichever of the two is longer. This may skew the metrics but it is arguably better
        # than including padding tokens in the calculation.
        for logit, label in zip(logits, labels):
            # Create a mask for this sample that excludes any padding tokens. The data collator can also
            # change the 'padding token' to -100 so we need to exclude both
            mask = (label != self.tokenizer.pad_token_id) & (label != -100)

            # Shift the label one position to the right because the model is predicting the
            # next token in the sequence and we want to compare the predicted token to the label.
            # Apply the mask to exclude any padding tokens
            label = label[1:][mask[1:]]
            prediction = logit[:-1][mask[:-1]]

            # If the prediction is longer than the label, truncate the prediction
            if len(prediction) > len(label):
                prediction = prediction[:len(label)]

            # If the label is longer than the prediction, truncate the label
            elif len(label) > len(prediction):
                label = label[:len(prediction)]

            # Append to the processed labels and predictions lists
            processed_labels.append(label)
            processed_predictions.append(prediction)

        # Return the processed labels and predictions. They are arrays of arrays. Each row represents
        # a label or logit (prediction). The number of columns is different because we have truncated
        # them each individually. So it is not a perfectly sized matrix.
        return (processed_predictions, processed_labels)

class F1Metric(Metric):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(
            metric_name='f1',
            compute_metrics=self.compute_f1_metrics,
            optimize_higher=True,
        )
        self.tokenizer = tokenizer
        # Load metrics once at init
        self.f1_metric= evaluate.load("f1")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.accuracy_metric = evaluate.load("accuracy")

    def compute_f1_metrics(self, eval_preds):
        logits, labels = eval_preds

        # Strip out padding tokens from the logits and labels and ensure each matching pair are the same length
        (predictions, references) = self._strip_padding(logits, labels)

        # Log the first 10 predictions and references pairs
        references_str = self.tokenizer.batch_decode(references, skip_special_tokens=True)
        predictions_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Log with the debug level the first 10 predictions and references pairs
        for i in range(min(10, len(references_str))):
            LOG.info(f"Reference {i}: {references_str[i]}")
            LOG.info(f"Prediction {i}: {predictions_str[i]}")

        # F1 metrics needs a single 1d array. So let's flatten them now
        references = np.concatenate(references).flatten()
        predictions = np.concatenate(predictions).flatten()

        f1_results = self.f1_metric.compute(predictions=predictions, references=references, average="weighted")["f1"]
        precision = self.precision_metric.compute(predictions=predictions, references=references, average="weighted", zero_division=0)["precision"]
        recall = self.recall_metric.compute(predictions=predictions, references=references, average="weighted", zero_division=0)["recall"]
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]

        return {"f1": f1_results, "precision": precision, "recall": recall, 'accuracy': accuracy}

class RougeMetric(Metric):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(
            metric_name='rougeL',
            compute_metrics=self.compute_rouge_metrics,
            optimize_higher=True,
        )
        self.tokenizer = tokenizer
        # Load metrics once at init
        self.rouge_metric = evaluate.load("rouge")
        self.f1_metric = evaluate.load("f1")

    def compute_rouge_metrics(self, eval_preds):
        logits, labels = eval_preds

        # Strip out padding tokens from the logits and labels and ensure each matching pair are the same length
        (predictions, references) = self._strip_padding(logits, labels)

        # Convert token ids to words for rouge calculation
        references_str = self.tokenizer.batch_decode(references, skip_special_tokens=True)
        predictions_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # F1 metrics needs a single 1d array. So let's flatten them now
        references_f1 = np.concatenate(references).flatten()
        predictions_f1 = np.concatenate(predictions).flatten()

        rouge_results = self.rouge_metric.compute(predictions=predictions_str, references=references_str, use_aggregator=True)
        f1_results = self.f1_metric.compute(predictions=predictions_f1, references=references_f1, average="weighted")

        # Combined both results datasets
        rouge_results.update(f1_results)

        return rouge_results

class BleuMetric(Metric):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(
            metric_name='bleu',
            compute_metrics=self.compute_bleu_metrics,
            optimize_higher=True,
        )
        self.tokenizer = tokenizer
        # Load metrics once at init
        self.metric = evaluate.load("bleu")

    def compute_bleu_metrics(self, eval_preds):
        logits, labels = eval_preds

        # Strip out padding tokens from the logits and labels and ensure each matching pair are the same length
        (predictions, references) = self._strip_padding(logits, labels)

        # Convert token ids to words
        references_str = self.tokenizer.batch_decode(references, skip_special_tokens=True)
        predictions_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        bleu_results = self.metric.compute(predictions=predictions_str, references=references_str)

        return bleu_results
