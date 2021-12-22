import json
import numpy as np
import pandas as pd
import tensorflow as tf


VOCAB_SIZE = 420

import os

Data = json.load(open('data/intent.json', "rb"))
Data = Data["intents"]
classes=[]
labels_and_features = []

for intent in Data:
    if(intent["tag"] not in classes):
        classes.append(intent["tag"])
    for feature in intent["patterns"]:
        labels_and_features.append((feature, intent["tag"]))


data_df = pd.DataFrame(data=labels_and_features)
data_df.columns = ["features", "labels"]
data_df["encoded_labels"] = data_df["labels"].astype('category').cat.codes


encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=20,
    input_shape=(1,)
    )

encoder.adapt(data_df["features"])


model = tf.keras.Sequential([
                          
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        

        # Use masking to handle the variable sequence lengths
      ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes))
])



model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])


history =model.fit(x=data_df["features"],y=data_df["encoded_labels"],epochs=100)

print(model.predict(["hello"]))
value =None
while(value not in ["y","n"] ):
    value = input("save the model y,n \n")

if(value=="y"):
    model.save("training/saved_model/chatbot",overwrite=True)


