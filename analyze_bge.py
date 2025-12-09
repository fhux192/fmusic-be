import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from collections import Counter
from tqdm import tqdm

DATA_FILE = 'data/song_dataset2.jsonl'
BASE_MODEL_ID = "BAAI/bge-m3"
LORA_PATH = "data/bge-m3-finetune/epoch_5"
BATCH_SIZE = 32

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

songs = []
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                songs.append(json.loads(line))
except FileNotFoundError:
    print(f"File {DATA_FILE} not found.")
    exit()

embedding_inputs = []
for song in songs:
    mood_str = ", ".join(song['mood_tags'])
    text = f"Mood: {mood_str}. Content: {song['semantic_text']}"
    embedding_inputs.append(text)

print("Loading Base Model & LoRA Adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModel.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

print("Generating embeddings based on Mood & Semantic...")

all_embeddings = []

for i in tqdm(range(0, len(embedding_inputs), BATCH_SIZE), desc="Encoding"):
    batch_texts = embedding_inputs[i: i + BATCH_SIZE]

    encoded_input = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    batch_emb = mean_pooling(model_output, encoded_input['attention_mask'])

    batch_emb = F.normalize(batch_emb, p=2, dim=1)

    all_embeddings.append(batch_emb.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0)

print("Clustering...")
NUM_CLUSTERS = 8
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
cluster_ids = kmeans.fit_predict(embeddings)

cluster_names = {}
for cluster_id in range(NUM_CLUSTERS):
    indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]
    all_tags = []
    for i in indices:
        all_tags.extend(songs[i]['mood_tags'])

    most_common = Counter(all_tags).most_common(2)
    if most_common:
        cluster_name = " & ".join([tag[0].capitalize() for tag in most_common])
    else:
        cluster_name = f"Cluster {cluster_id}"

    cluster_names[cluster_id] = f"{cluster_name} ({len(indices)} songs)"

readable_labels = [cluster_names[c_id] for c_id in cluster_ids]

print("Calculating coordinates for visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate=200)
embeddings_2d = tsne.fit_transform(embeddings)

df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'Cluster': readable_labels,
    'Title': [s['title'] for s in songs],
    'Artist': [s['artist'] for s in songs],
    'Moods': [", ".join(s['mood_tags']) for s in songs],
    'Semantic': [s['semantic_text'][:80] + "..." for s in songs]
})

print("Plotting Light Mode chart...")
fig = px.scatter(
    df,
    x='x', y='y',
    color='Cluster',
    hover_name='Title',
    hover_data={'x': False, 'y': False, 'Cluster': True, 'Artist': True, 'Moods': True, 'Semantic': True},
    title='Song Distribution by Mood & Semantic (Fine-tuned BGE-M3)',
    template='plotly_white',
    height=800
)

fig.update_traces(
    marker=dict(
        size=12,
        opacity=0.8,
        line=dict(
            width=1,
            color='Black'
        )
    )
)

fig.update_layout(
    legend_title_text='Main Mood Groups',
    font=dict(family="Arial", size=14, color="black"),
    plot_bgcolor='rgba(245, 245, 245, 1)'
)

output_file = "mood_clusters_finetuned.html"
fig.write_html(output_file, auto_open=True)
print(f"Done! Open '{output_file}' to view the chart.")