import os

# remember to set the env variable TESI_ROOT_PATH 
root_project_path = os.environ['TESI_PATH']


class _DataPath:
    base =                  root_project_path + '/data'
    mywikitables =          root_project_path + '/data/my_wikitables'
    wikitables =            root_project_path + '/data/turl_sloth_wikitables'


class _ModelPath:
    base =                  root_project_path + '/models'
    fasttext =              root_project_path + '/models/fastText'
    pre_trained_TaBERT =    root_project_path + '/models/pre-trained-TaBERT'


class _DBPath:
    base =                  root_project_path + '/db'
    chroma =                root_project_path + '/db/chroma'
    faiss =                 root_project_path + '/db/faiss'


class _JosieStuffPath:
    base =                  root_project_path + '/JOSIE'


class DefaultPath:
    root_project_path = root_project_path
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath
    josie_stuff_path = _JosieStuffPath

