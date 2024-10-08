import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction, get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
import numpy as np
import gc
from torch.optim import AdamW
import evaluate
import optuna

def load_data(file_path):
    sentences, labels = [], []
    sentence, label = [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            else:
                token, ner_label = line.strip().split('\t')
                sentence.append(token)
                label.append(ner_label)
    
    return sentences, labels

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sentences, train_labels = load_data(args.train)
    validation_sentences, validation_labels = load_data(args.validation)
    test_sentences, test_labels = load_data(args.test)
        
    all_labels = list(set([label for label_list in train_labels + validation_labels + test_labels for label in label_list]))
    label_map = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.use_local_model: 
        model = AutoModelForTokenClassification.from_pretrained(args.local_model_path, num_labels=len(label_map)).to(device)
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_map)).to(device)

    def tokenize_and_align_labels(sentences, labels):
        tokenized_inputs = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=128,
            is_split_into_words=True,
            return_tensors="pt"
        )

        label_all_tokens = True

        new_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_map[label[word_idx]])
                else:
                    label_ids.append(label_map[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            new_labels.append(label_ids)

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_train = tokenize_and_align_labels(train_sentences, train_labels)
    tokenized_validation = tokenize_and_align_labels(validation_sentences, validation_labels)
    tokenized_test = tokenize_and_align_labels(test_sentences, test_labels)

    def convert_to_dataset(tokenized_data, sentences, labels):
        input_ids = [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tokenized_data['input_ids']]
        attention_mask = [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tokenized_data['attention_mask']]
        label_ids = [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tokenized_data['labels']]

        data = [{'input_ids': input_ids[i], 'attention_mask': attention_mask[i], 'labels': label_ids[i], 'tokens': sentences[i], 'ner_tags': labels[i]} for i in range(len(sentences))]

        return Dataset.from_pandas(pd.DataFrame(data))

    dataset = DatasetDict({
        'train': convert_to_dataset(tokenized_train, train_sentences, train_labels),
        'validation': convert_to_dataset(tokenized_validation, validation_sentences, validation_labels),
        'test': convert_to_dataset(tokenized_test, test_sentences, test_labels)
    })

    def compute_metrics(p: EvalPrediction):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[id_to_label[id_] for id_ in label if id_ != -100] for label in labels]
        true_predictions = [[id_to_label[id_] for id_, label in zip(prediction, label) if label != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    metric = evaluate.load("seqeval")

    def objective(trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            logging_dir=args.logging_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            num_train_epochs=30.0,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type='linear',
            seed=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=["tensorboard"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=dataset['validation'])
        f1 = eval_result["eval_f1"]

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")

    best_learning_rate = trial.params["learning_rate"]
    print(f"Best learning rate: {best_learning_rate}")

    # Retrain with the best learning rate
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        logging_dir=args.logging_dir,
        learning_rate=best_learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=30.0,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        seed=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    # Calculate the number of training steps
    num_training_steps = len(dataset['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=best_learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # 10% of total steps for warmup
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    def save_model_contiguous(model, save_path):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        model.save_pretrained(save_path)

    trainer.train()

    results = trainer.evaluate(eval_dataset=dataset['test'])
    print("Result on test dataset")
    print(results)

    save_model_contiguous(model, args.save_pretrained)

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a token classification model.")
    
    parser.add_argument("--train", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--validation", type=str, required=True, help="Path to the validation data.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--tokenizer_name", type=str, required=False, help="Tokenizer name or path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model.")
    parser.add_argument("--logging_dir", type=str, required=True, help="Directory for logging.")
    parser.add_argument("--save_pretrained", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--use_local_model", type=bool, default=False, help="Whether to use a local model")
    parser.add_argument("--local_model_path", type=str, help="Path to the local model")
    
    args = parser.parse_args()
    
    if args.use_local_model and not args.local_model_path:
        parser.error("--local_model_path is required when --use_local_model is set to True")
    
    main(args)
