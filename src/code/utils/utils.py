import os
import re 
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# tr = str.maketrans('', '', string.punctuation.replace('-', '')) # to keep minus sign

try:
    nltk.data.find('stopwords', f"{os.environ['HOME']}/nltk_data")
except LookupError:
    nltk.download('stopwords')
    
stopwords_set = set(stopwords.words('english'))



def round_to(n, precision):
    if n >= 0 or n < 0:
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision
    else:
        return n


def round_to_05(n):
    return float(format(round_to(n, 0.05), ".2f"))


def rebuild_table(table):
    return pd.DataFrame(
        data=[
            [entry_data['text'] 
             for entry_data in entry]
            for entry in table['tableData']
        ],
        columns=table['tableHeaders'][0]
        )


def my_tokenizer(s: str, remove_numbers=False):
    s = str(s)
    if not remove_numbers:
        return [            
            x for x in re.findall(r'\b([a-zA-Z]+|\d{1}|\d{2}|\d{3}|\d{4})\b', s) 
            if x not in stopwords_set
        ]
    else:
        return [
            x for x in re.findall(r'[a-zA-Z]+', s)
            if x not in stopwords_set
        ]


def cosine_similarity(a1:np.array, a2:np.array):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
