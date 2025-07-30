# === source ~/lora-env/bin/activate ===
# === sudo chown -R $USER:$USER /media/ndesantis/PS22001/lora-general-decision ===
# === torchrun --nproc_per_node=2 train_general_decision_lora.py ===
# === tensorboard --logdir="/media/ndesantis/PS22001/lora-general-decision/runs" ===


import os
import time
import torch
import torch.distributed as dist
import signal
import sys
import re

from glob import glob
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

# === ENVIRONMENT SETUP ===
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === INIT DISTRIBUTED ===
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
rank = dist.get_rank()

# === PATHS ===
model_name = "codellama/CodeLlama-13b-hf"
data_path = "/home/ndesantis/Desktop/General-Decision-Lora-Builder/JSONL-Refinement-Engine/lora_general_decision_dataset_v1"
output_dir = "/media/ndesantis/PS22001/lora-general-decision"


# === TOKENIZER & DATASET ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_from_disk(data_path)

# === Gives a Max Length ===
lengths = [(len(tokenizer(p + " " + r)["input_ids"])) for p, r in zip(dataset["prompt"], dataset["response"])]
import matplotlib.pyplot as plt
plt.hist(lengths, bins=50)
plt.title("Prompt+Response Token Lengths")
plt.xlabel("Number of tokens")
plt.ylabel("Count")
plt.show()
print("Max length:", max(lengths))
print("Average length:", sum(lengths) // len(lengths))
print("95th percentile:", sorted(lengths)[int(0.95 * len(lengths))])
print("99th percentile:", sorted(lengths)[int(0.99 * len(lengths))])



def tokenize(batch):
    prompts = [p + " " + r for p, r in zip(batch["prompt"], batch["response"])]
    return tokenizer(prompts, truncation=True, padding="max_length", max_length=420)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# === DATALOADER ===
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
sampler = DistributedSampler(tokenized)
dataloader = DataLoader(
    tokenized,
    batch_size=3,
    sampler=sampler,
    collate_fn=collator,
    num_workers=8,
    pin_memory=True,
)

# === VAL DATALOADER ===
val_sampler = DistributedSampler(tokenized, shuffle=True)
val_dataloader = DataLoader(
    tokenized,
    batch_size=3,
    sampler=val_sampler,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True,
)


# === LOAD MODEL ===
if rank == 0:
    print("🔧 Loading model...")

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        use_safetensors=True,
        use_cache=False,
    )
except Exception as e:
    if rank == 0:
        print(f"❌ Failed to load model: {e}")
    dist.destroy_process_group()
    sys.exit(1)

base_model = base_model.to(device)

# === TARGET MODULES ===
if rank == 0:
    print("🔍 Finding compatible target modules...")

target_modules = []
for name, module in base_model.named_modules():
    if isinstance(module, torch.nn.Linear) and module.in_features == module.out_features:
        if rank == 0:
            print(f"✅ {name}: {module.in_features} → {module.out_features}")
        target_modules.append(name.split(".")[-1])

target_modules = list(set(target_modules))
if rank == 0:
    print("🎯 Final target_modules:", target_modules)
    print(base_model.config.hidden_size)

base_model.config.use_cache = False

# === APPLY LORA ===
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = model.to(device)
model.train()

# === OPTIMIZER ===
optimizer = AdamW(model.parameters(), lr=2e-4)

# === TensorBoad ===
from torch.utils.tensorboard import SummaryWriter

# 🛠️ Ensure the directory exists
os.makedirs(f"{output_dir}/runs", exist_ok=True)

writer = SummaryWriter(log_dir=f"{output_dir}/runs")


# === AUTOSCAN CHECKPOINT ===
start_epoch = 0
global_step = 0


def find_latest_valid_checkpoint(dirs):
    for ckpt_dir in sorted(dirs, key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)), reverse=True):
        full_state = os.path.join(ckpt_dir, "full_state.pt")
        try:
            torch.load(full_state, map_location="cpu", weights_only=False)
            return ckpt_dir  # Found the most recent valid one
        except Exception as e:
            if rank == 0:
                print(f"❌ Skipping {ckpt_dir}: {e}")
    return None

checkpoint_dirs = glob(os.path.join(output_dir, "checkpoint-*"))
latest_ckpt = find_latest_valid_checkpoint(checkpoint_dirs)
          

if latest_ckpt:
    checkpoint_path = os.path.join(latest_ckpt, "full_state.pt")
    if os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"🔄 Resuming from latest checkpoint: {latest_ckpt}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # ✅ DEBUG PRINT
        if rank == 0:
            print("🧠 Loaded checkpoint keys:", checkpoint.keys())

        # ✅ Load states BEFORE wrapping in DDP
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]

        if rank == 0:
            print(f"🔁 Resumed from step {global_step}, epoch {start_epoch}")
    else:
        if rank == 0:
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
else:
    if rank == 0:
        print("🆕 No checkpoints found — starting from scratch")

# ✅ Wrap in DDP only after checkpoint load (always do this once!)
model = DDP(model, device_ids=[local_rank])

import shutil

def save_checkpoint(model, tokenizer, optimizer, epoch, global_step, output_dir, rank, retry=False):
    ckpt_dir = f"{output_dir}/checkpoint-{global_step}"
    os.makedirs(ckpt_dir, exist_ok=True)

    if rank == 0:
        print(f"💾 Saving checkpoint at step {global_step}...")

    try:
        # Save model and tokenizer safely
        model.module.save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)

        tmp_path = os.path.join(ckpt_dir, "full_state.tmp.pt")
        final_path = os.path.join(ckpt_dir, "full_state.pt")

        state_dict = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
        }

        with open(tmp_path, "wb") as f:
            torch.save(state_dict, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)

        # ✅ Verify immediately
        try:
            torch.load(final_path, map_location="cpu", weights_only=False)
            if rank == 0:
                print(f"📦 Saved and verified checkpoint at step {global_step}")
        except Exception as verify_error:
            print(f"❌ Checkpoint verification failed: {verify_error}")
            shutil.rmtree(ckpt_dir)
            print(f"🧹 Corrupted checkpoint deleted: {ckpt_dir}")
            if not retry:
                print("🔁 Retrying save once...")
                save_checkpoint(model, tokenizer, optimizer, epoch, global_step, output_dir, rank, retry=True)
            else:
                print("❌ Retry failed. Skipping save.")

    except Exception as e:
        print(f"❌ Failed to save checkpoint at step {global_step}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)




# === TRAINING LOOP ===
if rank == 0:
    print("🚀 Starting training...")

epochs = 2
accum_steps = 5
optimizer.zero_grad()
t0 = time.time()

try:
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and step < (global_step % len(dataloader)):
                continue

            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | Time/step: {time.time() - t0:.2f}s")
                    t0 = time.time()

                    if global_step % 1000 == 0:  # Every 1000 steps
                        model.eval()
                        with torch.no_grad():
                            val_loss = 0
                            val_batches = 0
                            for val_batch in val_dataloader:
                                for k in val_batch:
                                    val_batch[k] = val_batch[k].to(device, non_blocking=True)
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    outputs = model(
                                        input_ids=val_batch["input_ids"],
                                        attention_mask=val_batch["attention_mask"],
                                        labels=val_batch["labels"]
                                    )
                                val_loss += outputs.loss.item()
                                val_batches += 1
                                if val_batches >= 10:  # Small sample
                                    break
                            avg_val_loss = val_loss / val_batches
                        model.train()

                        
                        elapsed = time.time() - t0
                        tokens_this_step = batch["input_ids"].numel()
                        tokens_per_sec = tokens_this_step / elapsed
                        # Approx. 1 A100 hour ≈ $1.32 — adjust for your setup
                        gpu_cost_per_hour = .15
                        cost_per_step = (elapsed / 3600) * gpu_cost_per_hour
                        writer.add_scalar("perf/approx_cost_usd", cost_per_step, global_step)
                        writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, global_step)


                        if rank == 0:
                            print(f"⚙️ {tokens_per_sec:.2f} tokens/sec")
                            writer.add_scalar("val/loss", avg_val_loss, global_step)
                            print(f"🧪 Validation Loss @ Step {global_step}: {avg_val_loss:.4f}")


                    if global_step % 250 == 0:
                        ckpt_dir = f"{output_dir}/checkpoint-{global_step}"
                        os.makedirs(ckpt_dir, exist_ok=True)

                        max_retries = 3
                        for attempt in range(1, max_retries + 1):
                            try:
                                model.module.save_pretrained(ckpt_dir, safe_serialization=True)
                                tokenizer.save_pretrained(ckpt_dir)

                                tmp_path = os.path.join(ckpt_dir, "full_state.tmp.pt")
                                final_path = os.path.join(ckpt_dir, "full_state.pt")

                                state_dict = {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "model_state_dict": model.state_dict(),
                                }

                                with open(tmp_path, "wb") as f:
                                    torch.save(state_dict, f)
                                    f.flush()
                                    os.fsync(f.fileno())

                                os.replace(tmp_path, final_path)

                                # 🔍 Verify checkpoint
                                try:
                                    torch.load(final_path, map_location="cpu", weights_only=False)
                                    print(f"📦 Saved and verified checkpoint at step {global_step}")
                                    break  # ✅ Success — exit retry loop
                                except Exception as e:
                                    print(f"❌ Checkpoint verification failed: {e}")
                                    import shutil
                                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                                    print(f"🧹 Corrupted checkpoint deleted: {ckpt_dir}")
                                    if attempt == max_retries:
                                        print("❌ Max retries reached. Checkpoint permanently failed.")
                                    else:
                                        print(f"🔁 Retrying save... ({attempt}/{max_retries})")
                                        time.sleep(3)

                            except Exception as e:
                                print(f"❌ Failed to save checkpoint at step {global_step}: {e}")
                                if attempt == max_retries:
                                    print("❌ Max retries reached. Giving up.")
                                else:
                                    print(f"🔁 Retrying save... ({attempt}/{max_retries})")
                                    time.sleep(3)


except KeyboardInterrupt:
    if rank == 0:
        print("⚠️ Interrupted manually. Cleaning up...")

finally:
    if rank == 0:
        print("✅ Final save and shutdown...")
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    dist.destroy_process_group()
    torch.cuda.empty_cache()
