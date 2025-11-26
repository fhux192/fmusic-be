import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import json
import faiss
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

HAPPY_QUERY_KEYWORDS = [
    'happy', 'wedding', 'joyful', 'celebratory', 'party', 'romantic',
    'smile', 'laughing', 'fun', 'excited', 'love'
]
SAD_QUERY_KEYWORDS = [
    'sad', 'lonely', 'breakup', 'heartbreak', 'regret', 'crying',
    'melancholic', 'alone', 'gloomy'
]

FORBIDDEN_MOODS_FOR_HAPPY_QUERY = {'sad', 'heartbreak', 'regret', 'lonely', 'breakup', 'melancholic'}
FORBIDDEN_MOODS_FOR_SAD_QUERY = {'happy', 'celebratory', 'joyful', 'upbeat', 'wedding', 'party', 'fun'}

class SongRetriever:
    def __init__(self, model_name, index_path, song_map_path):
        self.device = self._get_device()
        print(f"[Retriever] Using device: {self.device}")

        print("[Retriever] Loading BGE model...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[Retriever] BGE model loaded.")

        print(f"[Retriever] Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)
        print(f"[Retriever] FAISS index loaded. Contains {self.index.ntotal} songs.")

        print(f"[Retriever] Loading song data map from '{song_map_path}'...")
        with open(song_map_path, 'r', encoding='utf-8') as f:
            song_list = json.load(f)
            self.index_to_song_info = {i: song for i, song in enumerate(song_list)}
        print("[Retriever] Song data map loaded.")

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _determine_mood_filters(self, semantic_text):

        query_text = semantic_text.lower()

        is_happy_query = any(keyword in query_text for keyword in HAPPY_QUERY_KEYWORDS)

        is_sad_query = any(keyword in query_text for keyword in SAD_QUERY_KEYWORDS)

        if is_happy_query and not is_sad_query:
            print("[Retriever] Query looks happy. Filtering out sad songs.")
            return FORBIDDEN_MOODS_FOR_HAPPY_QUERY

        if is_sad_query and not is_happy_query:
            print("[Retriever] Query looks sad. Filtering out happy songs.")
            return FORBIDDEN_MOODS_FOR_SAD_QUERY

        print("[Retriever] Query is neutral. No mood filter applied.")
        return set()

    def search(self, semantic_text, k=10, threshold=0.45):
        if not semantic_text:
            return []

        print(f"[Retriever] Searching for query: '{semantic_text}' with threshold > {threshold} and max results k={k}")

        query_embedding = self.model.encode(
            semantic_text,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        )

        query_vector = query_embedding.cpu().numpy().reshape(1, -1)

        forbidden_moods = self._determine_mood_filters(semantic_text)

        k_candidates = max(k * 3, 30)
        scores, indices = self.index.search(query_vector, k_candidates)

        filtered_songs = []
        for i in range(k_candidates):
            song_index = indices[0][i]
            score = scores[0][i]

            if song_index == -1:
                continue

            if score >= threshold:
                song_info = self.index_to_song_info.get(song_index)

                if song_info:
                    raw_moods = (
                        song_info.get('mood_tags')
                        or song_info.get('mood_tags[]')
                        or song_info.get('mood')
                        or []
                    )
                    if isinstance(raw_moods, str):
                        song_moods = set(m.strip() for m in raw_moods.split(',') if m.strip())
                    else:
                        song_moods = set(raw_moods)

                    if song_moods.isdisjoint(forbidden_moods):
                        current_score = round(float(score), 4)

                        print(
                            f"  -> Match found: [Score: {current_score}] {song_info.get('artist')} - {song_info.get('title')}")

                        filtered_songs.append({
                            'score': current_score,
                            'title': song_info.get('title'),
                            'artist': song_info.get('artist'),
                            'url': song_info.get('url'),
                        })

                    else:
                        print(f"  -> REJECTED (Mood Filter): {song_info.get('artist')} - {song_info.get('title')}")

        final_results = filtered_songs[:k]

        print(f"[Retriever] Found {len(final_results)} results meeting the threshold.")
        return final_results