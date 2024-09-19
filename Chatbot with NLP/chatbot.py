
pip install --upgrade gradio

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import pickle

model = tf.keras.models.load_model('/content/chatbot_model.keras')

with open('/content/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
def preprocess_input(text):
    text_seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(text_seq, maxlen=50)

def predict_genre(user_input):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    return "Marvel" if np.round(prediction[0][0]) == 0 else "DC"

import gradio as gr

iface = gr.Interface(
    fn=predict_genre,
    inputs="text",
    outputs="text",
    title="MOVIE CATEGORY PREDICTOR ",
    description="Enter a movie name to predict its CATEGORY (Marvel or DC)."
)

iface.launch(share=True)
