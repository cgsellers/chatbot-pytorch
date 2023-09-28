import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#descargar una vez y comentar
#nltk.download('punkt')

stemmer= PorterStemmer()

#devuelve tokens de una frase
def tokenize(frase):
    return nltk.word_tokenize(frase)

#devuelve la raíz de una palabra
def stem(palabra):
    return stemmer.stem(palabra.lower())

#devuelve la bolsa de palabras, la voy a crear manualmente
def bag_of_words(frase_tokenizada, palabras):
    frase_tokenizada = [stem(p) for p in frase_tokenizada]

    #creo un array 'vacío' para agregar un 1 a todas las palabras en el corpus 'palabras' que coincidan con la frase tokenizada
    bag=np.zeros(len(palabras), dtype=np.float32)
    for index, p in enumerate (palabras):
        if p in frase_tokenizada:
            bag[index]=1.0

    return bag


