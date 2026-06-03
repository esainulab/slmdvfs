import time, argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math, os, csv

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="logs")
parser.add_argument("--run_id", type=str, default="1")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading DistilGPT-2 model and tokenizer...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

print("[INFO] Loading WikiText-2 (validation split)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = dataset["text"]
texts = [t for t in texts if len(t.strip()) > 0]

# Use first N samples to keep it lightweight
N = 100
texts = texts[:N]

results = []
total_loss = 0
count = 0

criterion = torch.nn.CrossEntropyLoss()

print(f"[INFO] Starting inference on {N} samples...")
for i, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    # Warm-up once (for first iteration)
    if i == 0:
        with torch.no_grad():
            _ = model(**inputs)

    #torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        loss = criterion(outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
                         labels[:, 1:].reshape(-1))
    #torch.cuda.synchronize()
    t1 = time.time()

    total_loss += loss.item()
    count += 1
    latency = t1 - t0
    results.append({"sample": i, "latency_s": latency, "loss": loss.item()})

avg_loss = total_loss / count
ppl = math.exp(avg_loss)

csv_path = os.path.join(args.outdir, f"run{args.run_id}_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample", "latency_s", "loss"])
    writer.writeheader()
    writer.writerows(results)

print(f"[RESULT] Average loss={avg_loss:.4f}, Perplexity={ppl:.2f}")
print(f"[RESULT] Logs saved to {csv_path}")
