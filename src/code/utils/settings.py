

_root_path = '/home/giovanni/unimore/TESI/src/'


class _DataPath:
    base =                  _root_path + 'data/'
    mywikitables =          _root_path + 'data/my_wikitables/'
    wikitables =            _root_path + 'data/wikitables/'


class _ModelPath:
    base =                  _root_path + 'models/'
    fasttext =              _root_path + 'models/fastText/'
    pre_trained_TaBERT =    _root_path + 'models/pre-trained-TaBERT/'


class _DBPath:
    base =                  _root_path + 'db/'
    chroma =                _root_path + 'db/chroma/'

class DefaultPath:
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath

