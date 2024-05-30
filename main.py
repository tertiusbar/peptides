import pandas as pd
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI
import json
import pickle
from keras.models import load_model
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CLASE CALIFICADA ApiInput
class ApiInput(BaseModel):
    ### ESCRIBA SU CÓDIGO AQUÍ ###
    features: str
    ### FIN DEL CÓDIGO ###

# CLASE CALIFICADA ApiOutput
class ApiOutput(BaseModel):
    ### ESCRIBA SU CÓDIGO AQUÍ ###
    forecast: str
    ### FIN DEL CÓDIGO ###

app = FastAPI()

model = load_model('model_conv1d.h5') # cargamos el modelo.
    
with open('labels.json', 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text])
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def vectorize(X,tokenizer):
    # Vectorización de X
    X_vect = tokenizer.texts_to_sequences(X)
    # Longitud vectorizada del primer elemento
    len(X_vect)
    # Dejamos los vectores de igual longitud
    # Adicionamos padding en caso de ser necesario
    max_length = 100
    X_vect = pad_sequences(X_vect, maxlen=max_length, padding='post')
    return X_vect

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    input_series = pd.Series(data.features)
    input_preprocess = input_series.apply(preprocess)
    input_vect = vectorize(input_preprocess,tokenizer)
    predictions = np.argmax(model.predict(input_vect),axis=1)
    prediction = ApiOutput(forecast=list(json_object.keys())[predictions[0]])
    return prediction