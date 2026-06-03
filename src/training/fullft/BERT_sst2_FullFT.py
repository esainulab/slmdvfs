#!/usr/bin/env python3
"""
Optimized BERT Full Fine-Tuning for GLUE Tasks
Based on fast original implementation
Supports: SST-2, QNLI, MRPC
"""

import os
import time
import argparse
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
import evaluate
import numpy as np

# Dataset configurations
DATASET_CONFIGS = {
    "sst2": {
        "task_name": "sst2",
        "num_labels": 2,
        "sentence_keys": ("sentence", None),
        "metric_name": "accuracy",
    },
    "qnli": {
        "task_name": "qnli",
        "num_labels": 2,
        "sentence_keys": ("question", "sentence"),
        "metric_name": "accuracy",
    },
    "mrpc": {
        "task_name": "mrpc",
        "num_labels": 2,
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_name": "accuracy",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="BERT Full Fine-Tuning for GLUE tasks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "qnli", "mrpc"],
        help="Dataset to use (sst2, qnli, or mrpc)",
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable fp16 (use fp32). Recommended for DeBERTa-v2 to avoid instability.")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--log_every_n_steps", type=int, default=None,
                        help="Alias for --logging_steps (kept for experiment scripts)")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--eval_every_epoch", action="store_true", default=False,
                        help="Run evaluation at the end of every epoch (default: only once after training)")
    parser.add_argument("--enable_tqdm", action="store_true", default=False,
                        help="Enable tqdm progress bars (default: disabled)")
    return parser.parse_args()

class HeartbeatCallback(TrainerCallback):
    def __init__(self, total_steps: int, every_steps: int):
        self.total_steps = int(total_steps)
        self.every_steps = max(1, int(every_steps))

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Training steps: {self.total_steps} (logging every {self.every_steps} steps)", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == 1 or step % self.every_steps == 0:
            total = self.total_steps if self.total_steps > 0 else "?"
            print(f"Step {step}/{total}", flush=True)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_every_n_steps is not None:
        args.logging_steps = args.log_every_n_steps
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    config = DATASET_CONFIGS[args.dataset]
    sentence1_key, sentence2_key = config["sentence_keys"]
    
    print("=" * 60)
    print(f"BERT Full Fine-Tuning - {args.dataset.upper()}")
    print("=" * 60)
    print(f"Dataset: {config['task_name']}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    print("=" * 60)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    raw_datasets = load_dataset("glue", config["task_name"])
    
    print(f"Train samples: {len(raw_datasets['train'])}")
    print(f"Validation samples: {len(raw_datasets['validation'])}")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # Preprocess function
    def preprocess_function(examples):
        if sentence2_key is None:
            # Single sentence (SST-2)
            texts = (examples[sentence1_key],)
        else:
            # Sentence pair (QNLI, MRPC)
            texts = (examples[sentence1_key], examples[sentence2_key])
        return tokenizer(*texts, truncation=True, max_length=args.max_length)
    
    # Tokenize datasets
    print("🔧 Tokenizing datasets...")
    processed = raw_datasets.map(preprocess_function, batched=True)
    
    # Rename label column
    if "label" in processed["train"].column_names:
        processed = processed.rename_column("label", "labels")
    
    eval_dataset = processed["validation"]
    
    # Load model
    print(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=config["num_labels"],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Metrics
    metric = evaluate.load("glue", config["task_name"])
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    # Data collator (IMPORTANT: This speeds up training significantly)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments (optimized for speed)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if args.eval_every_epoch else "no",
        save_strategy="no",  # No checkpointing
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,  # Less frequent logging
        logging_first_step=True,
        load_best_model_at_end=False,
        report_to="none",
        fp16=args.fp16,
        dataloader_num_workers=0,  # Single worker (faster on Jetson)
        gradient_checkpointing=False,  # No gradient checkpointing
        disable_tqdm=not args.enable_tqdm,
    )
    
    # Create trainer (NO callbacks for maximum speed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train with timing
    print("\n🚀 Starting training...")
    print("=" * 60)

    steps_per_epoch = math.ceil(len(processed["train"]) / max(1, args.batch_size))
    total_steps = int(steps_per_epoch * args.epochs)
    trainer.add_callback(HeartbeatCallback(total_steps=total_steps, every_steps=args.logging_steps))
    
    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()
    
    train_time = t1 - t0
    
    print("=" * 60)
    print("✅ Training complete!")
    print(f"Train wall time: {train_time:.2f} s")
    print("=" * 60)
    
    # Evaluate
    print("\n📊 Evaluating on validation set...")
    t2 = time.perf_counter()
    results = trainer.evaluate()
    t3 = time.perf_counter()
    
    eval_time = t3 - t2
    
    # Get validation metric (Trainer prefixes with "eval_")
    val_metric = results.get(f"eval_{config['metric_name']}", results.get(config['metric_name']))
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Method: Full Fine-Tuning")
    print(f"Train wall time: {train_time:.2f} s")
    print(f"Eval wall time: {eval_time:.2f} s")
    if val_metric is not None:
        print(f"Validation {config['metric_name']}: {val_metric:.4f}")
    else:
        print("Validation metric not found in evaluation results.")
    print(f"Validation loss: {results.get('eval_loss', 'N/A')}")
    print("Evaluation results:", results)
    print("=" * 60)

if __name__ == "__main__":
    main()
