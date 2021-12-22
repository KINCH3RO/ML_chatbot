import json
import numpy as np
import pandas as pd
import tensorflow as tf


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


model = tf.keras.models.load_model('training/saved_model/chatbot')
print("model_loaded")


def getResponse(sentence):
    prediction = np.array(model.predict([sentence]))
    result = prediction.argmax()
    result = data_df[data_df["encoded_labels"] == result]["labels"].iloc[0]
    
    dictResult = {
        'pattern':sentence,
        'pred_percentage': float(prediction.max()),
        'tag': result,
        'responses': responseDict[result]

    }
    return dictResult


if __name__ == '__main__':
  
    print(getResponse("time"))
