import fasttext
import fasttext.util

from tools.utils.settings import DefaultPath as dp

target_dim = 128

print('Loading...')
ft = fasttext.load_model(f'{dp.model_path.fasttext}/cc.en.300.bin')

print('Reducing...')
fasttext.util.reduce_model(ft, target_dim)

print('Saving...')
ft.save_model(f'{dp.model_path.fasttext}/cc.en.{target_dim}.bin')