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
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        except:
          return render_template('index.html', value="")

    wav_file_pre  = os.listdir("./audio")[0]
    wav_file_pre = f"{os.getcwd()}/audio/{wav_file_pre}"
    wav_file = convert(wav_file_pre)
    os.remove(wav_file_pre)
   
    x_test = extract_feature(wav_file)
    y_pred = model.predict(np.array([x_test]))
    os.remove(wav_file)
    
    return render_template('index.html', value=y_pred[0])
    print(y_pred)
    


if __name__ == "__main__":
    app.run(debug=True,threaded=True)
