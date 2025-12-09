import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Config:
    MODEL_NAME = "BAAI/bge-m3"
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    USE_FP16 = True
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    MARGIN = 0.3
    OUTPUT_DIR = "./bge-m3-finetuned"

config = Config()

if not os.path.exists('data.json'):
    print("⚠️ Không tìm thấy data.json, tạo dữ liệu mẫu...")
    dummy = [{"anchor": "test", "positive": "dung", "negative": "sai"}] * 100
    with open('data.json', 'w') as f: json.dump(dummy, f)

with open('data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

triplets = []
for item in json_data:
    if item.get("anchor") and item.get("positive") and item.get("negative"):
        triplets.append(item)

train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=SEED)
print(f"📊 Train: {len(train_triplets)} | Val: {len(val_triplets)}")

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        def tokenize(text):
            return self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )

        enc_a = tokenize(item["anchor"])
        enc_p = tokenize(item["positive"])
        enc_n = tokenize(item["negative"])

        return {
            "anchor_input_ids": enc_a["input_ids"].squeeze(0),
            "anchor_attention_mask": enc_a["attention_mask"].squeeze(0),
            "positive_input_ids": enc_p["input_ids"].squeeze(0),
            "positive_attention_mask": enc_p["attention_mask"].squeeze(0),
            "negative_input_ids": enc_n["input_ids"].squeeze(0),
            "negative_attention_mask": enc_n["attention_mask"].squeeze(0)
        }

print("⬇️ Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

model = AutoModel.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.gradient_checkpointing_enable()

model.enable_input_require_grads()

print("🔧 Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=["query", "key", "value", "dense"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = TripletDataset(train_triplets, tokenizer, config.MAX_LENGTH)
val_dataset = TripletDataset(val_triplets, tokenizer, config.MAX_LENGTH)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.sim = torch.nn.CosineSimilarity(dim=-1)
    def forward(self, a, p, n):
        return torch.relu(self.sim(a, n) - self.sim(a, p) + self.margin).mean()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, scaler):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    optimizer.zero_grad()

    for idx, batch in enumerate(progress):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', enabled=config.USE_FP16):
            out_a = model(input_ids=batch["anchor_input_ids"], attention_mask=batch["anchor_attention_mask"])
            out_p = model(input_ids=batch["positive_input_ids"], attention_mask=batch["positive_attention_mask"])
            out_n = model(input_ids=batch["negative_input_ids"], attention_mask=batch["negative_attention_mask"])

            emb_a = mean_pooling(out_a, batch["anchor_attention_mask"])
            emb_p = mean_pooling(out_p, batch["positive_attention_mask"])
            emb_n = mean_pooling(out_n, batch["negative_attention_mask"])

            loss = loss_fn(emb_a, emb_p, emb_n) / config.GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        current_loss = loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        total_loss += current_loss
        if idx % 10 == 0:
            progress.set_postfix({"loss": f"{current_loss:.4f}"})

    return total_loss / len(dataloader)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
num_steps = len(train_dataloader) // config.GRADIENT_ACCUMULATION_STEPS * config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * config.WARMUP_RATIO), num_steps)
scaler = torch.amp.GradScaler('cuda', enabled=config.USE_FP16)
loss_fn = TripletLoss(margin=config.MARGIN)

print("🚀 Starting Training...")
for epoch in range(config.EPOCHS):
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, scaler)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    model.save_pretrained(f"{config.OUTPUT_DIR}/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/epoch_{epoch+1}")

print("✅ Training Finished!")
os.system(f"zip -r bge_m3_final.zip {config.OUTPUT_DIR}")
print(f"File saved: bge_m3_final.zip")