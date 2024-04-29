import os

# remember to set the env variable TESI_ROOT_PATH 
_root_path = '/home/giovanni/unimore/tesi-magistrale/' # os.environ['TESI_SRC_PATH']
_src_path = _root_path + 'src/'

class _DataPath:
    base =                  _src_path + 'data/'
    mywikitables =          _src_path + 'data/my_wikitables/'
    wikitables =            _src_path + 'data/turl_sloth_wikitables/'


class _ModelPath:
    base =                  _src_path + 'models/'
    fasttext =              _src_path + 'models/fastText/'
    pre_trained_TaBERT =    _src_path + 'models/pre-trained-TaBERT/'


class _DBPath:
    base =                  _src_path + 'db/'
    chroma =                _src_path + 'db/chroma/'
    faiss =                 _src_path + 'db/faiss/'


class _JosieStuffPath:
    base =                  _root_path + 'JOSIE/'


class DefaultPath:
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath
    josie_stuff_path = _JosieStuffPath

