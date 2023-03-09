import unicodedata
import string
import glob
import os
import numpy as np
import nltk
import torch

cwd = os.getcwd()
all_letters = string.ascii_letters
n_letters = len(all_letters)
all_categories = []

def load_data():
    files = glob.glob(os.path.join(cwd, "./pytorch_tutorial_scripts/rnn/data/names/*.txt"))
    data = {}
    for file in files:
        category = file.split('/')[-1].split('.')[0]
        names = open(files[0], encoding='utf-8').read().strip().split('\n')
        data[category] = names
        all_categories.append(category)

    return data


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
)

def tokenize(text):
    tokens = list(map(lambda x: x, text))
    return tokens

def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor