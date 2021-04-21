import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
from flask import Flask,render_template,request
import pickle # to save model after training

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
## Configure upload location for audio
app.config['UPLOAD_FOLDER'] = "./audio"

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result
## Route for home page
@app.route('/')
def home():
    return render_template('index.html',value="")


## Route for results
@app.route('/results', methods = ['GET', 'POST'])
def results():

    if not os.path.isdir("./audio"):
        os.mkdir("audio")
    if request.method == 'POST':
        try:
            f = request.files['file']
            filename = f.filename
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            wav_file_pre  = os.listdir("./audio")[0]
            wav_file = f"{os.getcwd()}/audio/{wav_file_pre}" 
            features = np.array(extract_feature(wav_file, mfcc=True, chroma=True, mel=True).reshape(1, -1))

            y_pred = model.predict(features)         
            os.remove(wav_file)
            
            return render_template('index.html', value=y_pred[0])
           
        except:
          return render_template('index.html', value="")

    
    


if __name__ == "__main__":
    app.run()
