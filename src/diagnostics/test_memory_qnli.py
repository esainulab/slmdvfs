#!/usr/bin/env python3
"""
Quick memory test for QNLI N=256 batch sizes before the full sweep.
Tests both BERT-base (batch=128) and DeBERTa-xlarge (batch=16).

Usage:
    python3 test_memory_qnli.py
"""

import torch
import gc
import sys

PYTHON_BIN = sys.executable

CONFIGS = [
    {
        "label": "BERT-base",
        "model_name": "bert-base-uncased",
        "batch_size": 128,
        "max_length": 256,
    },
    {
        "label": "DeBERTa-xlarge",
        "model_name": "microsoft/deberta-v2-xlarge",
        "batch_size": 16,
        "max_length": 256,
    },
]


def get_gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0, 0


def test_config(label, model_name, batch_size, max_length):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"  model={model_name}, batch={batch_size}, max_length={max_length}")
    print(f"{'='*60}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU total memory: {total_gb:.1f} GB")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)
        model.train()

        # Simulate a QNLI batch (question + sentence pair)
        dummy_q = ["What is the capital of France?"] * batch_size
        dummy_s = ["Paris is the capital and most populous city of France."] * batch_size

        print(f"Tokenizing dummy batch (size={batch_size}, max_length={max_length})...")
        inputs = tokenizer(
            dummy_q, dummy_s,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)

        print("Running forward + backward pass...")
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

        loss.backward()

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n✅ PASSED")
            print(f"   Peak GPU memory: {peak_gb:.2f} GB / {total_gb:.1f} GB ({peak_gb/total_gb*100:.1f}%)")
            if peak_gb / total_gb > 0.95:
                print(f"   ⚠️  WARNING: Very close to OOM — consider reducing batch size")
        else:
            print(f"✅ PASSED (CPU only)")

        return True

    except torch.cuda.OutOfMemoryError:
        print(f"❌ OUT OF MEMORY — reduce batch size for {label}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        # Cleanup
        try:
            del model, inputs, labels, outputs, loss
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("QNLI Memory Test — N=256")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  No CUDA device found — running on CPU (memory checks won't reflect GPU)")
    else:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_gb:.1f} GB")

    results = {}
    for cfg in CONFIGS:
        ok = test_config(**cfg)
        results[cfg["label"]] = ok

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for label, ok in results.items():
        status = "✅ OK" if ok else "❌ OOM"
        print(f"  {label}: {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✅ All batch sizes fit in memory — safe to start experiment.")
    else:
        print("❌ At least one config failed — adjust batch sizes before running the sweep.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
