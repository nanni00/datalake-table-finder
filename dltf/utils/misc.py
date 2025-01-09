import re
import time
from functools import reduce
from string import whitespace, digits, punctuation, ascii_uppercase, ascii_lowercase

import mmh3
import pandas as pd

from dltf.sloth.sloth import sloth


whitespace_translator =     str.maketrans(whitespace, ' ' * len(whitespace))
digits_translator =         str.maketrans(digits, ' ' * len(digits))
punctuation_translator =    str.maketrans(punctuation, ' ' * len(punctuation))
lowercase_translator =      str.maketrans(ascii_uppercase, ascii_lowercase)


def clean_string(s, *translators):
    if len(translators) == 0:
        translators = [str.maketrans('\n|', '  ')]
    return reduce(lambda si, tr: str(si).translate(tr), translators, str(s)).strip()


def get_string_translator(tr):
    match tr:
        case 'whitespace':  return whitespace_translator
        case 'digits':      return digits_translator
        case 'punctuation': return punctuation_translator
        case 'lowercase':   return lowercase_translator
        case _:            raise ValueError(f'Unknown translator: {tr}')


def mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def get_local_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")


def convert_to_giga(x):
    if x.endswith('GB'):
        return int(re.match(r'\d+', x).group())
    elif x.endswith('MB'):
        return int(re.match(r'\d+', x).group()) / 1024
    elif x.endswith('KB'):
        return int(re.match(r'\d+', x).group()) / (1024 ** 2)



def largest_overlap_sloth(table1, table2, valid_cols1, valid_cols2, verbose=False, blacklist=[], **sloth_args) -> tuple[int, float]:
    num_null = 0

    def format_value_for_excluding_nan(t):
        nonlocal num_null
        if not t or pd.isna(t) or t in blacklist:
            num_null += 1
            return f'{t}@{num_null}'
        t = clean_string(t)
        return t
    
    table1 = [[format_value_for_excluding_nan(row[i]) for row in table1] for i in range(len(table1[0])) if valid_cols1[i] == 1]
    table2 = [[format_value_for_excluding_nan(row[i]) for row in table2] for i in range(len(table2[0])) if valid_cols2[i] == 1]

    metrics = []
    start_sloth = time.time()
    try:
        results, metrics = sloth(table1, table2, metrics=metrics, verbose=verbose, **sloth_args)
        if results == []:
            return -2, round(time.time() - start_sloth, 3)
        largest_ov_sloth = metrics[-2]
        return largest_ov_sloth, round(time.time() - start_sloth, 3)
    except TimeoutError:
        return -1, round(time.time() - start_sloth, 3)
    except IndexError:        
        return -2, round(time.time() - start_sloth, 3)



def numerize(number):
    """
    A slightly different version of numerize() function from numerize_denumerize Python package
    from Dheeresh Agarwal, https://pypi.org/project/numerize-denumerize/
    """
    if not isinstance(number, (int, float)):
        raise ValueError("Input must be a number.")
    
    suffixes = {
        42: 'D',
        39: 'N',
        36: 'O',
        33: 'S',
        30: 'F',
        27: 'U',
        24: 'Y',
        21: 'Z',
        18: 'E',
        15: 'P',
        12: 'T',
        9:  'B',
        6:  'M',
        3:  'K'
    }

    for suffix in suffixes:
        if number >= 10**suffix:
            number /= 10**suffix
            return f"{round(number)}{suffixes[suffix]}"
    return round(number)



def chunks(sequence, chunk_size, *args):
    # Chunks of chunk_size documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield (sequence[j:j + chunk_size], *args)

