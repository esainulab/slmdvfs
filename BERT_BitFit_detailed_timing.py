#!/usr/bin/env python3
"""
BERT BitFit with Detailed Per-Step Timing + Event Logging
Logs each phase (Forward, Backward, Optimizer) for tegrastats labeling
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
    """Log events with timestamps to CSV for tegrastats labeling"""
    def __init__(self, output_path):
        self.output_path = output_path
        self.events = []
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        self.log_event("script_start", "Script started")
    
    def log_event(self, phase, description=""):
        current_time = time.time()
        event = {
            'timestamp': datetime.fromtimestamp(current_time).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3],
            'unix_timestamp': current_time,
            'elapsed_seconds': current_time - self.start_time if self.start_time else 0,
            'phase': phase,
            'description': description,
        }
        self.events.append(event)
        print(f"[{event['timestamp']}] {phase}: {description}")
    
    def save(self):
        if not self.events:
            return
        fieldnames = ['timestamp', 'unix_timestamp', 'elapsed_seconds', 'phase', 'description']
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
        print(f"\n✅ Events log saved to: {self.output_path}")

class StepTimer:
    """Track time for each phase of training step"""
    def __init__(self):
        self.data_loading = []
        self.forward = []
        self.backward = []
        self.optimizer = []
        self.total = []
    
    def record(self, phase, duration):
        getattr(self, phase).append(duration)
    
    def get_averages(self):
        return {
            'data_loading': np.mean(self.data_loading) if self.data_loading else 0,
            'forward': np.mean(self.forward) if self.forward else 0,
            'backward': np.mean(self.backward) if self.backward else 0,
            'optimizer': np.mean(self.optimizer) if self.optimizer else 0,
            'total': np.mean(self.total) if self.total else 0,
        }
    
    def reset(self):
        self.data_loading.clear()
        self.forward.clear()
        self.backward.clear()
        self.optimizer.clear()
        self.total.clear()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "qnli", "mrpc"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    return parser.parse_args()

def freeze_all_except_bias(model):
    """Freeze all parameters except bias terms (BitFit)"""
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch, 
                    step_timer, logger, log_every_n):
    """Train for one epoch with detailed timing and event logging"""
    model.train()
    total_loss = 0
    num_steps = len(train_loader)
    
    logger.log_event(f"epoch_{epoch+1}_training_start", f"Starting epoch {epoch+1} training")
    
    progress_bar = tqdm(total=num_steps, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(train_loader):
        step_start = time.perf_counter()
        
        # Log step phases (every N steps to avoid too many events)
        if step % log_every_n == 0 or step == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_data_loading", "Data loading phase")
        
        # PHASE 1: Data Loading
        data_start = time.perf_counter()
        batch = {k: v.to(device) for k, v in batch.items()}
        data_time = time.perf_counter() - data_start
        
        # PHASE 2: Forward Pass
        if step % log_every_n == 0 or step == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_forward", "Forward pass phase")
        
        forward_start = time.perf_counter()
        outputs = model(**batch)
        loss = outputs.loss
        forward_time = time.perf_counter() - forward_start
        
        # PHASE 3: Backward Pass
        if step % log_every_n == 0 or step == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_backward", "Backward pass phase")
        
        backward_start = time.perf_counter()
        loss.backward()
        backward_time = time.perf_counter() - backward_start
        
        # PHASE 4: Optimizer Step
        if step % log_every_n == 0 or step == 0:
            logger.log_event(f"epoch_{epoch+1}_step_{step}_optimizer", "Optimizer phase")
        
        optimizer_start = time.perf_counter()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        optimizer_time = time.perf_counter() - optimizer_start
        
        step_time = time.perf_counter() - step_start
        
        # Record timings
        step_timer.record('data_loading', data_time)
        step_timer.record('forward', forward_time)
        step_timer.record('backward', backward_time)
        step_timer.record('optimizer', optimizer_time)
        step_timer.record('total', step_time)
        
        total_loss += loss.item()
        
        # Log progress
        if (step + 1) % log_every_n == 0:
            avg_timings = step_timer.get_averages()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fwd': f'{avg_timings["forward"]*1000:.0f}ms',
                'bwd': f'{avg_timings["backward"]*1000:.0f}ms',
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    avg_loss = total_loss / num_steps
    logger.log_event(f"epoch_{epoch+1}_training_end", f"Epoch {epoch+1} training complete")
    
    return avg_loss

def evaluate_model(model, eval_loader, device, metric, logger, epoch):
    """Evaluate the model"""
    logger.log_event(f"epoch_{epoch+1}_evaluation_start", "Starting evaluation")
    
    model.eval()
    
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    results = metric.compute()
    logger.log_event(f"epoch_{epoch+1}_evaluation_end", "Evaluation complete")
    
    return results

def main():
    print("Script started")
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
    print(f"BERT BitFit - Detailed Timing - {args.dataset.upper()}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Events log: {events_log_path}")
    print("=" * 80)
    
    # Load dataset
    logger.log_event("data_loading_start", "Loading dataset")
    raw_datasets = load_dataset("glue", config["task_name"])
    logger.log_event("data_loading_end", "Dataset loaded")
    
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
    
    # Rename and clean
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
    logger.log_event("model_loading_start", "Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=config["num_labels"],
    )
    
    # Apply BitFit: freeze all except bias
    logger.log_event("bitfit_freezing_start", "Freezing non-bias parameters")
    freeze_all_except_bias(model)
    logger.log_event("bitfit_freezing_end", "BitFit applied")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.log_event("model_loading_end", "Model loaded")
    
    # Setup optimizer and scheduler
    logger.log_event("optimizer_setup_start", "Setting up optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.log_event("optimizer_setup_end", "Optimizer ready")
    
    # Setup metric
    metric = evaluate.load("glue", config["task_name"])
    
    # Training loop
    logger.log_event("training_start", f"Starting {args.epochs} epochs")
    
    step_timer = StepTimer()
    epoch_times = []
    
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        step_timer.reset()
        
        # Train one epoch
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler, device,
            epoch, step_timer, logger, args.log_every_n_steps
        )
        
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        
        # Get average timings
        avg_timings = step_timer.get_averages()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary")
        print(f"{'='*80}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f} s")
        print(f"\nPer-Step Timing Breakdown (averages):")
        print(f"  Data loading:    {avg_timings['data_loading']*1000:6.2f} ms ({avg_timings['data_loading']/avg_timings['total']*100:5.1f}%)")
        print(f"  Forward pass:    {avg_timings['forward']*1000:6.2f} ms ({avg_timings['forward']/avg_timings['total']*100:5.1f}%)")
        print(f"  Backward pass:   {avg_timings['backward']*1000:6.2f} ms ({avg_timings['backward']/avg_timings['total']*100:5.1f}%)")
        print(f"  Optimizer step:  {avg_timings['optimizer']*1000:6.2f} ms ({avg_timings['optimizer']/avg_timings['total']*100:5.1f}%)")
        print(f"  Total per step:  {avg_timings['total']*1000:6.2f} ms")
        print(f"  Steps per second: {1.0/avg_timings['total']:.2f}")
        print(f"{'='*80}\n")
        
        # Evaluate
        results = evaluate_model(model, eval_loader, device, metric, logger, epoch)
        print(f"Validation {config['metric_name']}: {results[config['metric_name']]:.4f}\n")
    
    logger.log_event("training_end", "All training complete")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Method: BitFit")
    print(f"Total epochs: {args.epochs}")
    print(f"Per-epoch times: {', '.join([f'{t:.2f}s' for t in epoch_times])}")
    print(f"Validation {config['metric_name']}: {results[config['metric_name']]:.4f}")
    print("=" * 80)
    
    logger.log_event("script_end", "Script complete")
    print("About to save events")
    logger.save()
    
    print(f"\n💡 Next: Run tegrastats with this events_log.csv to label each phase")

if __name__ == "__main__":
    main()