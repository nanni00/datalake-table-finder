import re
import time
from typing import Literal 
import numpy as np
import pandas as pd
import polars as pl


def print_info(**dec_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            on_line = 'on_line' in dec_kwargs and dec_kwargs['on_line'] == True
            end_line = '\r' if on_line else '\n'
            
            msg_before = dec_kwargs['msg_before'] if 'msg_before' in dec_kwargs else '' 
            msg_after = dec_kwargs['msg_after'] if 'msg_after' in dec_kwargs else ''
            msg_after = msg_after if not on_line else f'{msg_before} {msg_after}' if msg_before else msg_after 
            
            if msg_before: 
                print(msg_before, end=end_line)
            start = time.time()
            results = func(*args, **kwargs)            
            end = time.time()

            if 'time' in dec_kwargs: 
                print(f'Elapsed time: {round(end - start, 3)}s')
            if msg_after:
                print(msg_after)
            return results
        return wrapper
    return decorator


def round_to(n, precision):
    if n >= 0 or n < 0:
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision
    else:
        return n


def round_to_05(n):
    return float(format(round_to(n, 0.05), ".2f"))


def rebuild_table(table, mode:Literal['pandas', 'polars', 'polars.lazy', 'text']='pandas'):
    if mode == 'pandas': 
        return pd.DataFrame(
            data=[
                [entry_data['text'] 
                 for entry_data in entry]
                for entry in table['tableData']
            ],
            columns=table['tableHeaders'][0]
            )
    elif mode == 'polars':
        return pl.DataFrame(
            data=[
                [entry_data['text'] 
                 for entry_data in entry]
                for entry in table['tableData']
            ],
            schema=table['tableHeaders'][0]
            )
    elif mode == 'polars.lazy':
        s = table['tableHeaders'][0]
        s = [c if s.count(c) == 1 else f'{c}#{i}' for i, c in enumerate(s)]
        return pl.LazyFrame(
            data=[
                [entry_data['text'] 
                 for entry_data in entry]
                for entry in table['tableData']
            ],
            schema=s #table['tableHeaders'][0]
            )
    elif mode == 'text':
        header = '\t'.join([str(h) for h in table['tableHeaders'][0]])
        data = '\n'.join(['\t'.join(str(entry_data['text']) for entry_data in entry) for entry in table['tableData']])
        return f'{header}\n{data}'
    else:
        raise ValueError(f'Unknown mode: {mode}')


def my_tokenizer(s: str, remove_numbers=False):
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words('english'))

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


def cosine_similarity(arr1:np.array, arr2:np.array):
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))


def get_int_from_(s: str):
    return [int(x) for x in re.findall(r'\d+', s)]
    

def get_current_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")

