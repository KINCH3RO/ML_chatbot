import json
import numpy as np
import pandas as pd
import tensorflow as tf
import random


VOCAB_SIZE = 420

Data = json.load(open('data/intent.json', "rb"))
Data = Data["intents"]
classes = []
labels_and_features = []
responseDict = {}

for intent in Data:
    if(intent["tag"] not in classes):
        classes.append(intent["tag"])
    for feature in intent["patterns"]:
        labels_and_features.append((feature, intent["tag"]))
    responseDict[intent["tag"]] = intent["responses"]


data_df = pd.DataFrame(data=labels_and_features)
data_df.columns = ["features", "labels"]
data_df["encoded_labels"] = data_df["labels"].astype('category').cat.codes


model = tf.keras.models.load_model('training/saved_model_python/chatbot')





question="fdgdsgs"
while(len(question) != 0):
    question = input()
    result = np.array(model.predict([question])).argmax()
    result = data_df[data_df["encoded_labels"] == result]["labels"].iloc[0]
    print(result)
