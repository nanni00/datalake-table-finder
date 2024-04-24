import os

# remember to set the env variable TESI_ROOT_PATH 
_root_path = '/home/giovanni/unimore/tesi-magistrale/src/' # os.environ['TESI_SRC_PATH']


class _DataPath:
    base =                  _root_path + 'data/'
    mywikitables =          _root_path + 'data/my_wikitables/'
    wikitables =            _root_path + 'data/turl_sloth_wikitables/'


class _ModelPath:
    base =                  _root_path + 'models/'
    fasttext =              _root_path + 'models/fastText/'
    pre_trained_TaBERT =    _root_path + 'models/pre-trained-TaBERT/'


class _DBPath:
    base =                  _root_path + 'db/'
    chroma =                _root_path + 'db/chroma/'
    faiss =                 _root_path + 'db/faiss/'



class DefaultPath:
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath

