import json
import faiss
from sentence_transformers import SentenceTransformer
import torch
import os
import re

BGE_MODEL_NAME = "BAAI/bge-m3"

SONG_DATA_PATH = "data/song_dataset2.jsonl"
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


def extract_youtube_id(url):
    if not url:
        return None
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def convert_to_embed_url(original_url):
    video_id = extract_youtube_id(original_url)
    if video_id:
        return f"https://www.youtube.com/embed/{video_id}"
    return original_url


def load_songs(filepath):
    songs = []
    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                songs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Cảnh báo: Lỗi định dạng JSON ở dòng {i + 1}")
    return songs


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print(f"Loading model '{BGE_MODEL_NAME}'...")
    # Load model
    model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
    print("Model loaded.")

    print(f"Reading song data from '{SONG_DATA_PATH}'...")
    songs = load_songs(SONG_DATA_PATH)

    if not songs:
        print("Không có dữ liệu bài hát nào. Kết thúc.")
        return

    # Lấy text ngữ nghĩa để embedding
    corpus = [song.get('semantic_text', '') for song in songs]
    print(f"Read {len(corpus)} songs.")

    print("Starting embedding process...")
    song_embeddings = model.encode(
        corpus,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Embedding complete. Shape: {song_embeddings.shape}")

    embedding_dim = song_embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dim)

    print("Adding embeddings to FAISS index...")
    index.add(song_embeddings)
    print(f"Done. Index now contains {index.ntotal} vectors.")

    print(f"Saving index to '{INDEX_SAVE_PATH}'...")
    faiss.write_index(index, INDEX_SAVE_PATH)

    print("Processing metadata and converting URLs...")
    song_data_map = []
    for song in songs:
        embed_url = convert_to_embed_url(song.get('url', ''))

        song_data_map.append({
            'id': song.get('song_id'),
            'title': song.get('title'),
            'artist': song.get('artist'),
            'original_url': song.get('url'),
            'url': embed_url,
            'semantic_text': song.get('semantic_text')
        })

    print(f"Saving song data map to '{SONG_MAP_SAVE_PATH}'...")
    with open(SONG_MAP_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(song_data_map, f, ensure_ascii=False, indent=4)

    print("\n✅ Preparation process complete! Vector DB is ready.")


if __name__ == "__main__":
    main()