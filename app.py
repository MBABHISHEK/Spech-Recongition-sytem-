import os
from flask import Flask, render_template, request, redirect, url_for
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pydub import AudioSegment

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def convert_mp3_to_wav(mp3_file):
    # Load MP3 file using pydub
    audio = AudioSegment.from_mp3(mp3_file)
    # Export as WAV
    wav_file = mp3_file[:-4] + '.wav'  # Assuming mp3_file ends with '.mp3'
    audio.export(wav_file, format="wav")
    return wav_file

def summarize_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    freq_table = dict()

    for word in words:
        word = word.lower()
        if word in stop_words:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    sentences = sent_tokenize(text)
    sentence_value = dict()

    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq
                else:
                    sentence_value[sentence] = freq

    sum_values = 0
    for sentence in sentence_value:
        sum_values += sentence_value[sentence]

    average = int(sum_values / len(sentence_value))

    summary = ''
    for sentence in sentences:
        if (sentence in sentence_value) and (sentence_value[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file to the configured upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Convert MP3 to WAV
        wav_file = convert_mp3_to_wav("uploads/marketplace.mp3")

        # Perform speech recognition
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            summary = summarize_text(text)
            return render_template('result.html', original_text=text, summary=summary)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            return f"Error processing audio: {e}"

if __name__ == '__main__':
    app.run(debug=True)
