#!/usr/bin/env python3
"""
BERT BitFit with Event Logging
Logs all phases to events_log.csv with timestamps for tegrastats overlay
Only bias parameters are fine-tuned (BitFit method)
"""

import os
import time
import argparse
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)
import evaluate
from tqdm.auto import tqdm

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

class EventLogger:
    """Log events with timestamps to CSV"""
    def __init__(self, output_path):
        self.output_path = output_path
        self.events = []
        self.start_time = None
    
    def start(self):
        """Mark the absolute start time"""
        self.start_time = time.time()
        self.log_event("script_start", "Script execution started")
    
    def log_event(self, phase, description="", metadata=None):
        """Log an event with timestamp"""
        current_time = time.time()
        
        event = {
            'timestamp': datetime.fromtimestamp(current_time).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3],
            'unix_timestamp': current_time,
            'elapsed_seconds': current_time - self.start_time if self.start_time else 0,
            'phase': phase,
            'description': description,
        }
        
        if metadata:
            event.update(metadata)
        
        self.events.append(event)
        
        # Print to console too
        print(f"[{event['timestamp']}] {phase}: {description}")
    
    def save(self):
        """Save all events to CSV"""
        if not self.events:
            return
        
        # Get all unique keys
        all_keys = set()
        for event in self.events:
            all_keys.update(event.keys())
        
        fieldnames = ['timestamp', 'unix_timestamp', 'elapsed_seconds', 'phase', 'description']
        # Add any extra metadata fields
        extra_fields = sorted(all_keys - set(fieldnames))
        fieldnames.extend(extra_fields)
        
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
        
        print(f"\n✅ Events log saved to: {self.output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "qnli", "mrpc"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch, logger, log_every_n):
    """Train for one epoch with event logging"""
    model.train()
    total_loss = 0
    num_steps = len(train_loader)
    
    logger.log_event(f"epoch_{epoch+1}_start", f"Starting epoch {epoch+1}/{args.epochs}", 
                     {'epoch': epoch+1, 'num_steps': num_steps})
    
    progress_bar = tqdm(total=num_steps, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(train_loader):
        step_start = time.time()
        
        # Log step start (every N steps to avoid too many events)
        if step % log_every_n == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_start", 
                           f"Step {step}/{num_steps}",
                           {'epoch': epoch+1, 'step': step, 'phase_detail': 'step_start'})
        
        # Data to GPU
        data_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        data_time = time.time() - data_start
        
        # Forward pass
        forward_start = time.time()
        outputs = model(**batch)
        loss = outputs.loss
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer_start = time.time()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start
        
        step_time = time.time() - step_start
        total_loss += loss.item()
        
        # Log step end (every N steps)
        if step % log_every_n == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_end",
                           f"Step {step} complete",
                           {
                               'epoch': epoch+1,
                               'step': step,
                               'phase_detail': 'step_end',
                               'loss': float(loss.item()),
                               'step_time_ms': step_time * 1000,
                               'data_time_ms': data_time * 1000,
                               'forward_time_ms': forward_time * 1000,
                               'backward_time_ms': backward_time * 1000,
                               'optimizer_time_ms': optimizer_time * 1000,
                           })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    avg_loss = total_loss / num_steps
    logger.log_event(f"epoch_{epoch+1}_end", f"Epoch {epoch+1} complete",
                     {'epoch': epoch+1, 'avg_loss': avg_loss})
    
    return avg_loss

def evaluate_model(model, eval_loader, device, metric, logger):
    """Evaluate with event logging"""
    logger.log_event("evaluation_start", "Starting validation evaluation")
    
    model.eval()
    
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    results = metric.compute()
    
    logger.log_event("evaluation_end", "Validation evaluation complete",
                     {'accuracy': float(results.get('accuracy', 0))})
    
    return results

def main():
    global args
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DATASET_CONFIGS[args.dataset]
    sentence1_key, sentence2_key = config["sentence_keys"]
    
    # Initialize event logger
    events_log_path = os.path.join(args.output_dir, "events_log.csv")
    logger = EventLogger(events_log_path)
    logger.start()
    
    print("=" * 80)
    print(f"BERT BitFit with Event Logging - {args.dataset.upper()}")
    print("=" * 80)
    print(f"Events will be logged to: {events_log_path}")
    print("=" * 80)
    
    # Load dataset
    logger.log_event("data_loading_start", "Loading dataset from HuggingFace")
    raw_datasets = load_dataset("glue", config["task_name"])
    logger.log_event("data_loading_end", "Dataset loaded",
                     {'train_samples': len(raw_datasets['train']),
                      'val_samples': len(raw_datasets['validation'])})
    
    # Load tokenizer
    logger.log_event("tokenizer_loading_start", "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    logger.log_event("tokenizer_loading_end", "Tokenizer loaded")
    
    # Preprocess
    def preprocess_function(examples):
        if sentence2_key is None:
            texts = (examples[sentence1_key],)
        else:
            texts = (examples[sentence1_key], examples[sentence2_key])
        return tokenizer(*texts, truncation=True, max_length=args.max_length)
    
    logger.log_event("tokenization_start", "Tokenizing datasets")
    processed = raw_datasets.map(preprocess_function, batched=True)
    logger.log_event("tokenization_end", "Tokenization complete")
    
    # Rename label -> labels FIRST, then remove unused columns
    if "label" in processed["train"].column_names:
        processed = processed.rename_column("label", "labels")
    
    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [col for col in processed["train"].column_names if col not in keep_cols]
    if remove_cols:
        processed = processed.remove_columns(remove_cols)
    
    processed.set_format("torch")
    
    # Create dataloaders
    logger.log_event("dataloader_creation_start", "Creating dataloaders")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(
        processed["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    eval_loader = DataLoader(
        processed["validation"],
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        num_workers=0,
    )
    logger.log_event("dataloader_creation_end", "Dataloaders created")
    
    # Load model
    logger.log_event("model_loading_start", f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=config["num_labels"],
    )
    model.to(device)
    
    # BitFit: Freeze all parameters except biases
    logger.log_event("bitfit_freeze_start", "Applying BitFit: freezing non-bias parameters")
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    logger.log_event("bitfit_freeze_end", "BitFit applied: non-bias parameters frozen",
                     {'total_params': total_params,
                      'frozen_params': frozen_params,
                      'trainable_params': trainable_params,
                      'device': str(device)})
    
    # Setup optimizer and scheduler (only on trainable params)
    logger.log_event("optimizer_setup_start", "Setting up optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.log_event("optimizer_setup_end", "Optimizer and scheduler ready",
                     {'total_steps': num_training_steps,
                      'warmup_steps': num_warmup_steps})
    
    # Setup metric
    metric = evaluate.load("glue", config["task_name"])
    
    # Training loop
    logger.log_event("training_start", f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler, device,
            epoch, logger, args.log_every_n_steps
        )
        
        # Evaluate after each epoch
        results = evaluate_model(model, eval_loader, device, metric, logger)
    
    logger.log_event("training_end", "Training complete")
    
    # Final evaluation
    logger.log_event("final_evaluation_start", "Final evaluation on validation set")
    final_results = evaluate_model(model, eval_loader, device, metric, logger)
    logger.log_event("final_evaluation_end", "Final evaluation complete",
                     {'final_accuracy': float(final_results.get('accuracy', 0))})
    
    logger.log_event("script_end", "Script execution complete")
    
    # Save events log
    logger.save()
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Method: BitFit")
    print(f"Validation {config['metric_name']}: {final_results[config['metric_name']]:.4f}")
    print(f"\nEvents log saved to: {events_log_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()