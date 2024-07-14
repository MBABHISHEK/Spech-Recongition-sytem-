from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import os
import torch
import librosa
import numpy as np
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Initialize Translator
translator = Translator()

# Route to upload audio file
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
        else:
            return jsonify({'error': 'No file part in the request'}), 400
    return render_template('upload.html')

# Route to transcribe audio
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file_path = request.json.get('filename')
    try:
        # Read audio file
        data = wavfile.read(file_path)
        framerate = data[0]
        sounddata = data[1]
        input_audio, _ = librosa.load(file_path, sr=16000)
        input_values = tokenizer(input_audio, return_tensors="pt").input_values

        # Perform transcription
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

        # Return transcription result
        response = jsonify({'transcription': transcription})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Set CORS headers
        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
