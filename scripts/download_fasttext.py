"""
Simple utils to download and compress the fastText model
"""

import os
import shutil
import fasttext
import fasttext.util


def download_fasttext(ft_model_path, language='en'):
    fasttext.util.download_model(language, if_exists='ignore')
    shutil.move(f'./cc.{language}.300.bin', ft_model_path)


def compress_fasttext(src_ft_model_path, dst_ft_model_path, to_dim):
    ft = fasttext.load_model(src_ft_model_path)
    fasttext.util.reduce_model(ft, to_dim)
    ft.save_model(dst_ft_model_path)


if __name__ == '__main__':
    # create default directory 
    if not os.path.exists(f'{os.environ["DLTFPATH"]}/models/fasttext'):
        os.makedirs(f'{os.environ["DLTFPATH"]}/models/fasttext')

    language = 'en'
    ft_model_path = f'{os.environ["DLTFPATH"]}/models/fasttext/cc.{language}.300.bin'
    print(f'Downloading {language.upper()} language fastText model...')
    download_fasttext(ft_model_path, language)
    os.remove(f'{ft_model_path}.gz')
    print('Done.')

    to_dim = 128
    reduced_ft_model_path = f'{os.environ["DLTFPATH"]}/models/fasttext/cc.{language}.{to_dim}.bin'
    print(f'Compressing fastText model to dimensionality {to_dim}...')
    compress_fasttext(ft_model_path, reduced_ft_model_path, to_dim)
    print('Done.')
