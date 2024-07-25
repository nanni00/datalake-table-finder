import os

# remember to set the env variable TESI_ROOT_PATH 
root_project_path = os.environ['THESIS_PATH']


class _DataPath:
    base =                  root_project_path + '/data'
    tests =                 root_project_path + '/data/tests'


class _ModelPath:
    base =                  root_project_path + '/models'
    fasttext =              root_project_path + '/models/fasttext'
    tabert =                root_project_path + '/models/TaBERT'


class _DBPath:
    base =                  root_project_path + '/db'
    chroma =                root_project_path + '/db/chroma'
    faiss =                 root_project_path + '/db/faiss'


class DefaultPath:
    root_project_path = root_project_path
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath

