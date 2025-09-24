
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model_handler import generate_description_from_image
from retriever import SongRetriever

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
INDEX_PATH = "vector_db/faiss_index.bin"
SONG_MAP_PATH = "vector_db/song_data_map.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Initializing Song Retriever...")
song_retriever = SongRetriever(
    model_name=BGE_MODEL_NAME,
    index_path=INDEX_PATH,
    song_map_path=SONG_MAP_PATH
)
print("Song Retriever is ready.")

@app.route('/suggest', methods=['POST'])
def suggest_song_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        description = generate_description_from_image(filepath)
        os.remove(filepath)  

        if description:
            DEFAULT_K = 10 
            k_str = request.args.get('top_k', str(DEFAULT_K))
            try:
                k = int(k_str)
                if not (1 <= k <= 20):  
                    k = DEFAULT_K
            except ValueError:
                k = DEFAULT_K

            DEFAULT_THRESHOLD = 0.35
            threshold_str = request.args.get('threshold', str(DEFAULT_THRESHOLD))
            try:
                threshold = float(threshold_str)
                if not (0.3 <= threshold <= 0.9): 
                    threshold = DEFAULT_THRESHOLD
            except ValueError:
                threshold = DEFAULT_THRESHOLD
                
            print(f"Searching for songs with score >= {threshold}, max results: {k}")

            suggested_songs = song_retriever.search(
                description, 
                k=k,
                threshold=threshold
            )
            
            return jsonify({
                'description': description,
                'suggested_songs': suggested_songs
            })
        else:
            return jsonify({'error': 'Could not process the image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


# ---

# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from model_handler import generate_description_from_image
# from retriever import SongRetriever

# app = Flask(__name__)
# CORS(app)

# BGE_MODEL_NAME = "BAAI/bge-m3"
# UPLOAD_FOLDER = 'uploads'
# INDEX_PATH = "vector_db/faiss_index.bin"
# SONG_MAP_PATH = "vector_db/song_data_map.json"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# print("Initializing Song Retriever...")
# song_retriever = SongRetriever(
#     model_name=BGE_MODEL_NAME,
#     index_path=INDEX_PATH,
#     song_map_path=SONG_MAP_PATH
# )
# print("Song Retriever is ready.")

# @app.route('/suggest', methods=['POST'])
# def suggest_song_route():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         description = generate_description_from_image(filepath)
#         os.remove(filepath)  

#         if description:
#             query_with_instruction = f"Represent this sentence for searching relevant passages: {description}"
            
#             DEFAULT_K = 10 
#             k_str = request.args.get('top_k', str(DEFAULT_K))
#             try:
#                 k = int(k_str)
#                 if not (1 <= k <= 20): k = DEFAULT_K
#             except ValueError:
#                 k = DEFAULT_K

#             DEFAULT_THRESHOLD = 0.35
#             threshold_str = request.args.get('threshold', str(DEFAULT_THRESHOLD))
#             try:
#                 threshold = float(threshold_str)
#                 if not (0.3 <= threshold <= 0.9): threshold = DEFAULT_THRESHOLD
#             except ValueError:
#                 threshold = DEFAULT_THRESHOLD
                
#             print(f"Searching for songs with score >= {threshold}, max results: {k}")

#             suggested_songs = song_retriever.search(
#                 query_with_instruction, 
#                 k=k,
#                 threshold=threshold
#             )
            
#             return jsonify({
#                 'description': description,
#                 'suggested_songs': suggested_songs
#             })
#         else:
#             return jsonify({'error': 'Could not process the image'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)