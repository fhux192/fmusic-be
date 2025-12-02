import json
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from collections import Counter

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data_file = 'data.jsonl'
songs = []
try:
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                songs.append(json.loads(line))
except FileNotFoundError:
    print(f"File {data_file} not found.")
    exit()

embedding_inputs = []
for song in songs:
    mood_str = ", ".join(song['mood_tags'])
    # Translated the prefix text to English
    text = f"Mood: {mood_str}. Content: {song['semantic_text']}"
    embedding_inputs.append(text)

print("Generating embeddings based on Mood & Semantic...")
model = SentenceTransformer('BAAI/bge-m3', device=device)
embeddings = model.encode(embedding_inputs, convert_to_tensor=False, normalize_embeddings=True)

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

    # Translated label format
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
    title='Song Distribution by Mood & Semantic (BGE-M3)',
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

output_file = "mood_clusters_light.html"
fig.write_html(output_file, auto_open=True)
print(f"Done! Open '{output_file}' to view the chart.")