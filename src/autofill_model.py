import pandas as pd
import numpy as np
import evaluate
import logging
import os.path
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
    AutoConfig,
    EvalPrediction
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('[Autofill Model]')

warnings.filterwarnings('ignore')

INPUT_FEATURES = [
    'html_cleaned',
    'label',
    'ml_dataset',
    'language'
]

BEST_HYPERPARAMS = {
    'learning_rate': 0.000082,
    'num_train_epochs': 12,
    'weight_decay': 0.1,
    'per_device_train_batch_size': 32,
}

DEFAULT_MODEL_PATH = 'google/mobilebert-uncased'
DEFAULT_DATA_PATH = 'vazish/autofill_dataset'
DEFAULT_SAVE_PATH = './artifacts'

NEGATIVE_LABEL_ID = 16

class AutofillModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        learning_rate: float = BEST_HYPERPARAMS['learning_rate'],
        batch_size: int = BEST_HYPERPARAMS['per_device_train_batch_size'],
        weight_decay: float = BEST_HYPERPARAMS['weight_decay'],
        epochs: int = BEST_HYPERPARAMS['num_train_epochs'],
        seed: int = 42,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        self.pipe = None
        self.save_path = None
        self.trainer = None

    def load_data(self, path: str = DEFAULT_DATA_PATH):
        logger.info(f'Loading data from : {path}\n')
        self.dataset = load_dataset(path)

    def _compute_metrics(self, eval_preds: EvalPrediction):
        # load metrics
        f1_metric = evaluate.load('f1')
        precision_metric = evaluate.load('precision')
        recall_metric = evaluate.load('recall')

        # predictions
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)

        # compute metrics
        precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')['precision']
        recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')['recall']
        f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')['f1']
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def _save_model(self, push_to_hub: bool = False):
        logger.info(f'Saving tokenizer and model to: {self.save_path}')
        label2id = self.dataset['train'].features["labels"]._str2int
        id2label = {v: k for k, v in id2label.items()}

        self.tokenizer.save_pretrained(self.save_path)
        config = AutoConfig.from_pretrained(self.save_path, label2id=label2id, id2label=id2label)
        model = AutoModelForSequenceClassification.from_pretrained(self.save_path, config=config)
        model.save_pretrained(self.save_path)

        if push_to_hub:
            # store on hugging face hub, requires login with huggingface-cli login
            self.trainer.push_to_hub(path)

    def _load_model(self, path: str):
        self.pipe = pipeline('text-classification', model=path)

    def train(self, save_path=DEFAULT_SAVE_PATH):
        self.save_path = save_path
        # Get the classifier
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.dataset['train'].features["labels"].names)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenized_datasets = self.dataset.map(
            lambda dataset: self.tokenizer(dataset['html_cleaned'], truncation=True), batched=True
        )

        # only keep necessary columns for trainer
        self.tokenized_datasets.remove_columns(['html_cleaned', 'ml_dataset', 'language'])
        # Training args
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            self.model_name,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            save_strategy="no",
            report_to="none",
            evaluation_strategy="epoch",
            logging_strategy='epoch',
        )

        self.trainer = Trainer(
            self.classifier,
            training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )
        self.trainer.train()
        logger.info('Done Training \n')
        self._save_model(save_path)
        self.evaluate()

    def predict(self, html_tags: list[str]):
        if self.pipe is None:
            self.pipe = pipeline('text-classification', model=self.save_path, truncation=True, batch_size=8)
        return self.pipe(html_tags)

    def evaluate():
        def pprint_metrics(y_true, y_preds, with_cm=False):
            logger.info(f'Precision: {precision_score(y_true, y_preds, average="weighted", zero_division=0)}')
            logger.info(f'Recall: {recall_score(y_true, y_preds, average="weighted", zero_division=0)}')
            logger.info(f'F1: {f1_score(y_true, y_preds, average="weighted", zero_division=0)}')
            if with_cm:
                # log confusion matrix
                logger.info('\n')
                logger.info(confusion_matrix(y_true, y_preds))
                # log the classification report
                logger.info('\n')
                logger.info(
                    classification_report(
                        y_true,
                        y_preds,
                        digits=3,
                        zero_division=0,
                    )
                )
            return

        # Make predictions on test set
        logger.info('Testing metrics... \n')
        class_label = ds['test'].features['labels'].int2str
        actuals = class_label(ds['test']['labels'])
        predictions = self.predict(ds['html_cleaned'])
        pprint_metrics(actuals, predictions, with_cm=True)
        return


# def train(save_path):
#     autofill_model = AutofillModel()
#     data_path = './data/dataset.parquet'
#     autofill_model.load_data(data_path, features=INPUT_FEATURES, from_cache_if_existing=True)
#     autofill_model.train(save_path)
#
#
# def predict(load_path):
#     autofill_model = AutofillModel()
#     autofill_model.load_model(path=load_path)
#     texts = [
#         '<input autocomplete="new-password" class="_1dHn9" data-hook="wsr-input" maxlength="524288" style="text-overflow: clip;" type="password" value=""/>',
#         '<input checked="" class="rs-input creditcard" data-label="Carte de crédit" data-tr-customized-input="true" name="zaart" style="display: none;" type="radio" value=""/>',
#         '<meta content="Thu, 26 Jan 2023 15:30:37 GMT" http-equiv="Memento-Datetime"/>',
#     ]
#     predictions = [{'text': t, 'label': autofill_model.predict(t)} for t in texts]
#     print(predictions)
#
#
# def evaluate_model(load_path):
#     autofill_model = AutofillModel(model_name=load_path)
#     autofill_model.load_model(path=load_path)
#     data_path = ''  # we're using cache here
#     autofill_model.load_data(data_path, features=INPUT_FEATURES, from_cache_if_existing=True)
#     autofill_model.evaluate()
