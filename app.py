import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
import gensim.downloader
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
import json

nltk.download('punkt')

app = FastAPI()

# Set up logging
logging.basicConfig(filename='interaction_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Cache the GloVe model
@st.cache_resource
def load_glove_model():
    return gensim.downloader.load('glove-wiki-gigaword-100')

# Load GloVe model
glove_model = load_glove_model()

# Function to get word embeddings
def get_word_embeddings(sentence, glove_model):
    embeddings = []
    for word in word_tokenize(sentence.lower()):
        if word in glove_model:
            embeddings.append(glove_model[word])
        else:
            embeddings.append(np.zeros(glove_model.vector_size))
    return embeddings

# Define CNN Model Class
class CNNModel(Model):
    def __init__(self, sequence_length):
        super(CNNModel, self).__init__()
        self.sequence_length = sequence_length
        self.conv1d_layer1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
        self.conv1d_layer2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.global_max_pooling = layers.GlobalMaxPooling1D()
        self.dense = layers.Dense(sequence_length * 4, activation='softmax')

    def call(self, inputs):
        x = self.conv1d_layer1(inputs)
        x = self.conv1d_layer2(x)
        x = self.global_max_pooling(x)
        x = self.dense(x)
        return tf.reshape(x, (-1, self.sequence_length, 4))

label_mapping = {
    0: 'B-AC',
    1: 'B-LF',
    2: 'B-O',
    3: 'I-LF'
}

cnn_model = CNNModel(sequence_length=50)
cnn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.load_weights('cnn_glove_shubh.weights.h5')

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    embeddings = get_word_embeddings(text, glove_model)
    max_sequence_length = 50

    if not embeddings:
        return {"error": "No embeddings generated for the input text."}

    pca = PCA(n_components=2, svd_solver='full')
    embeddings_pca = pca.fit_transform(embeddings)
    embeddings_pca_padded = pad_sequences([embeddings_pca], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
    embeddings_array = np.array(embeddings_pca_padded)

    predictions = cnn_model.predict(embeddings_array)
    predicted_indices = np.argmax(predictions, axis=2)[0][:len(embeddings)]
    predicted_labels = [label_mapping[index] for index in predicted_indices]

    # Log the interaction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": text,
        "predicted_labels": predicted_labels
    }
    logging.info(json.dumps(log_entry))

    return {"predictions": predicted_labels}

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8503)

st.title("CNN Model with GloVe for Sequence Labelling")
st.write("Input text for Named Entity Recognition")

# Input text
user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    embeddings = get_word_embeddings(user_input, glove_model)
    max_sequence_length = 50

    if not embeddings:
        st.write("No embeddings generated for the input text.")
    else:
        pca = PCA(n_components=2, svd_solver='full')
        embeddings_pca = pca.fit_transform(embeddings)
        embeddings_pca_padded = pad_sequences([embeddings_pca], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
        embeddings_array = np.array(embeddings_pca_padded)

        predictions = cnn_model.predict(embeddings_array)
        predicted_indices = np.argmax(predictions, axis=2)[0][:len(embeddings)]
        predicted_labels = [label_mapping[index] for index in predicted_indices]

        st.write("Predicted Labels:", predicted_labels)

        # Log the interaction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "predicted_labels": predicted_labels
        }
        logging.info(json.dumps(log_entry))

if __name__ == "__main__":
    run_fastapi()

