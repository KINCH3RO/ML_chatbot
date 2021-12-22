from flask import Flask
from gtts import gTTS
import os
import ML_bot
from flask import send_file
app = Flask(__name__)

@app.route("/api/getPrediction/<keyword>")
def getResponse(keyword):
    return ML_bot.getResponse(keyword)


@app.route("/api/getAudioText/<keyword>")
def getAudioClip(keyword):
    save_path ="static/output.mp3"
    tts = gTTS(keyword)
    tts.save(save_path)
    return send_file(os.getcwd()+"/"+save_path)