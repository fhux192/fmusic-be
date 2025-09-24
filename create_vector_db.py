
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import os

BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SONG_DATA_PATH = "data/song_dataset.jsonl"
VECTOR_DB_DIR = "vector_db"
INDEX_SAVE_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
SONG_MAP_SAVE_PATH = os.path.join(VECTOR_DB_DIR, "song_data_map.json")

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def load_songs(filepath):
    songs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            songs.append(json.loads(line))
    return songs

def main():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print("Loading BGE model...")
    model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
    print("BGE model loaded.")

    print(f"Reading song data from '{SONG_DATA_PATH}'...")
    songs = load_songs(SONG_DATA_PATH)
    corpus = [song['semantic_text'] for song in songs]
    print(f"Read {len(corpus)} songs.")

    print("Starting embedding process. This may take a few minutes...")
    song_embeddings = model.encode(
        corpus,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Embedding complete. Shape of embeddings: {song_embeddings.shape}")

    embedding_dim = song_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    print("Adding embeddings to FAISS index...")
    index.add(song_embeddings)
    print(f"Done. Index now contains {index.ntotal} vectors.")

    print(f"Saving index to '{INDEX_SAVE_PATH}'...")
    faiss.write_index(index, INDEX_SAVE_PATH)

    song_data_map = [
        {
            'id': song['song_id'],
            'title': song['title'],
            'artist': song['artist'],
            'genre': song['genre'],
            'url': f'https://www.youtube.com/embed/0y-4pEJtuPo'
        }
        for song in songs
    ]

    print(f"Saving song data map to '{SONG_MAP_SAVE_PATH}'...")
    with open(SONG_MAP_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(song_data_map, f, ensure_ascii=False, indent=4)

    print("\nPreparation process complete!")

if __name__ == "__main__":
    main()

# ---

# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import torch
# import os

# BGE_MODEL_NAME = "BAAI/bge-m3"
# SONG_DATA_PATH = "data/song_dataset.jsonl"
# VECTOR_DB_DIR = "vector_db"
# INDEX_SAVE_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
# SONG_MAP_SAVE_PATH = os.path.join(VECTOR_DB_DIR, "song_data_map.json")

# def get_device():
#     if torch.backends.mps.is_available():
#         return "mps"
#     elif torch.cuda.is_available():
#         return "cuda"
#     else:
#         return "cpu"

# DEVICE = get_device()
# print(f"Using device: {DEVICE}")

# def load_songs(filepath):
#     songs = []
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             songs.append(json.loads(line))
#     return songs

# def main():
#     os.makedirs(VECTOR_DB_DIR, exist_ok=True)

#     print(f"Loading BGE model: {BGE_MODEL_NAME}...")
#     model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
#     print("BGE model loaded successfully.")

#     print(f"Reading song data from '{SONG_DATA_PATH}'...")
#     songs = load_songs(SONG_DATA_PATH)
#     corpus = [song['semantic_text'] for song in songs]
#     print(f"Read {len(corpus)} songs.")

#     print("Starting embedding process. This may take a few minutes...")
#     song_embeddings = model.encode(
#         corpus,
#         batch_size=32,        
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True 
#     )
#     print(f"Embedding complete. Shape of embeddings: {song_embeddings.shape}")

#     embedding_dim = song_embeddings.shape[1]
#     index = faiss.IndexFlatIP(embedding_dim) 
#     print("Adding embeddings to FAISS index...")
#     index.add(song_embeddings)
#     print(f"Done. Index now contains {index.ntotal} vectors.")

#     print(f"Saving index to '{INDEX_SAVE_PATH}'...")
#     faiss.write_index(index, INDEX_SAVE_PATH)

#     song_data_map = [
#         {
#             'id': song['song_id'],
#             'title': song['title'],
#             'artist': song['artist'],
#             'genre': song['genre'],
#             'url': f'https://www.youtube.com/embed/0y-4pEJtuPo' 
#         }
#         for song in songs
#     ]

#     print(f"Saving song data map to '{SONG_MAP_SAVE_PATH}'...")
#     with open(SONG_MAP_SAVE_PATH, 'w', encoding='utf-8') as f:
#         json.dump(song_data_map, f, ensure_ascii=False, indent=4)

#     print("\nPreparation process complete!")
#     print(f"Vector DB and song map have been saved in '{VECTOR_DB_DIR}' directory.")

# if __name__ == "__main__":
#     main()