import random
import json

import torch

from modelo import RedNeuronal
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#leer el corpus una vez más y cargarlo en pytorch
with open('corpus.json', 'r') as json_data:
    corpus = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)

#declarar las variables guardadas en el entrenamiento para cargar el modelo que ya habíamos entrenado
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
palabras = data['palabras']
tags = data['tags']
model_state = data["model_state"]

model = RedNeuronal(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

nombre_bot = "Agorín"
print(f'Bienvenido a la página web del IES Agora, soy {nombre_bot}, para terminar la conversacion escribe "salir"')
#INICIO DE LA CONVERSACIÓN
while True:
    frase = input("usuario: ")
    if frase == "salir":
        break

    #con el input hacer una predicción de intención con nuestro modelo
    frase = tokenize(frase)
    X = bag_of_words(frase, palabras)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    #escoger la intención que más probabilidades tenga según la predicción
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probabilidades = torch.softmax(output, dim=1)
    prob = probabilidades[0][predicted.item()]

    #Si la predicción más probable supera el 65%, enviarla al chat, si no, no damos un output
    if prob.item() > 0.65:
        for intencion in corpus['intenciones']:
            if tag == intencion["tag"]:
                print(f"{nombre_bot}: {random.choice(intencion['respuestas'])}")
    else:
        print(f"{nombre_bot}: Disculpa, no entiendo...")