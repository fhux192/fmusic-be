# file: retriever.py

import json
import faiss
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class SongRetriever:
    def __init__(self, model_name, index_path, song_map_path):
        self.device = self._get_device()
        print(f"[Retriever] Using device: {self.device}")
        
        print("[Retriever] Loading BGE model...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[Retriever] BGE model loaded.")
        
        print(f"[Retriever] Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)
        self.all_song_embeddings = torch.from_numpy(self.index.reconstruct_n(0, self.index.ntotal)).to(self.device)
        print(f"[Retriever] FAISS index loaded and embeddings cached. Contains {self.index.ntotal} songs.")
        
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

    def search(self, semantic_text, k=10, threshold=0.55):
        if not semantic_text:
            return []
            
        print(f"[Retriever] Searching for query: '{semantic_text}' with threshold > {threshold} and max results k={k}")
        
        query_with_instruction = f"Represent this sentence for searching relevant passages: {semantic_text}"

        query_embedding = self.model.encode(
            query_with_instruction, 
            convert_to_tensor=True,
            device=self.device
        )
        
        k_candidates = max(k * 3, 30)
        hits = util.semantic_search(
            query_embedding, 
            self.all_song_embeddings, 
            top_k=k_candidates
        )[0]

        filtered_songs = []
        for hit in hits:
            if hit['score'] >= threshold:
                song_index = hit['corpus_id']
                song_info = self.index_to_song_info.get(song_index)
                
                if song_info:
                    filtered_songs.append({
                        'score': round(hit['score'], 4), 
                        'title': song_info.get('title'),
                        'artist': song_info.get('artist'),
                        'url': song_info.get('url'),
                        'id': song_info.get('id')
                    })
        
        final_results = filtered_songs[:k]
                
        print(f"[Retriever] Found {len(final_results)} results meeting the threshold.")
        return final_results