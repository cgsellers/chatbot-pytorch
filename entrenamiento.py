import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from modelo import RedNeuronal

with open('corpus.json', 'r') as f:
    intenciones = json.load(f)

palabras = []
tags = []
xy = []
# vamos a cargar los datos de nuestro corpus: palabras, intenciones, tags...
for intencion in intenciones['intenciones']:
    tag = intencion['tag']
    tags.append(tag)
    #tokenizar las palabras de cada patrón del corpus
    for palabra in intencion['patrones']:
        #p:palabra tokenizada
        p = tokenize(palabra)
        palabras.extend(p)
        # añadir la tupla de (token y tag) a nuestra lista de tuplas
        xy.append((p, tag))

#filtrar signos
filtro_signoss = ['?', '.', '!']
# vamos a procesar las palabras en una lista de comprensión para tener la raiz en minúsculas (lower()) 
palabras = [stem(w) for w in palabras if w not in filtro_signoss]
# quitar duplicados gracias a los sets
palabras = sorted(set(palabras))
tags = sorted(set(tags))

##COMPROBAR QUE SE HAN PROCESADO LOS TEXTOS CORRECTAMENTE
#print(len(xy), "patrones")
#print(len(tags), "tags:", tags)
#print(len(palabras), "palabras:", palabras)


X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, palabras)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Declarar los hiperparámetros de nuestro modelo
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


#declarar nuestro dataset personalizado partiendo de Dataset de pytorch
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

#utilizar la aceleración por hardware siempre que sea necesario, como en google Colab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


modelo = RedNeuronal(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

# ENTRENAMIENTO
for epoch in range(num_epochs):
    for (palabra, etiqueta) in train_loader:
        #pasarle las palabras y etiquetas al hardware declarado anteriormente
        palabra = palabra.to(device)
        etiqueta = etiqueta.to(dtype=torch.long).to(device)
        
        outputs = modelo(palabra)
        loss = criterion(outputs, etiqueta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": modelo.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"palabras": palabras,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Entrenamiento completado. Archivo guardado en {FILE}')
