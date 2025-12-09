import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

def init_app():
    from model_handler import generate_description_from_image
    from retriever import SongRetriever
    return generate_description_from_image, SongRetriever

app = Flask(__name__)
CORS(app)

BGE_MODEL_NAME = "data/bge-m3-finetune/epoch_5"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
INDEX_PATH = "vector_db/faiss_index.bin"
SONG_MAP_PATH = "vector_db/song_data_map.json"
DEFAULT_K = 30
DEFAULT_THRESHOLD = 0.5
MAX_K = 30
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.9
MAX_UPLOAD_SIZE = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

generate_description_from_image = None
song_retriever = None

if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    app.logger.info("Initializing in worker process...")
    generate_description_from_image, SongRetriever = init_app()
    app.logger.info("Initializing Song Retriever...")
    song_retriever = SongRetriever(
        model_name=BGE_MODEL_NAME,
        index_path=INDEX_PATH,
        song_map_path=SONG_MAP_PATH
    )
    app.logger.info("Song Retriever is ready.")
else:
    app.logger.info("Skipping init in reloader parent process.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/suggest', methods=['POST'])
def suggest_song_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or file type not allowed'}), 400
    filepath = None
    try:
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f"Processing image: {filepath}")
        description = generate_description_from_image(filepath)
        if not description or len(description.strip()) < 10:
            app.logger.error("Generated description is empty or too short.")
            return jsonify({'error': 'Could not generate a valid description from the image'}), 500
        app.logger.info(f"Generated description: '{description}'")
        k = request.args.get('top_k', default=DEFAULT_K, type=int)
        if not (1 <= k <= MAX_K):
            k = DEFAULT_K
        threshold = request.args.get('threshold', default=DEFAULT_THRESHOLD, type=float)
        if not (MIN_THRESHOLD <= threshold <= MAX_THRESHOLD):
            threshold = DEFAULT_THRESHOLD
        app.logger.info(f"Searching for songs with score >= {threshold}, max results: {k}")
        suggested_songs = song_retriever.search(
            description,
            k=k,
            threshold=threshold
        )
        return jsonify({
            'description': description,
            'suggested_songs': suggested_songs
        })
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                app.logger.info(f"Removed temporary file: {filepath}")
            except Exception as e:
                app.logger.warning(f"Failed to remove temporary file {filepath}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)