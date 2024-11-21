# Data Lake Table Finder

This repository contains scripts and files to compare different methods in data discovery tasks.

## Setup

1. Download the repository

2. Create a python environment (see environment.yml)

3. Create the project environment variable and add it to the PYTHONPATH
```
export DLTFPATH=/path/to/datalake-table-finder
export PYTHONPATH=$PYTHONPATH:$DLTHPATH
```

4. Load the datalake on MongoDB, or store it as a local file (e.g. SantosLarge). In both cases, a numeric index and a field which describes the valid columns of each table must be created. For these tasks, check the Python script at script/load_data.py.

5. If you want to use JOSIE or MATE, you have to create a database that will store their index structures. The postgresql.conf file is the same used in the original JOSIE repository.

## Testing

Once you've completed the setup, to run tests create JSON configuration files, which provide all the details needed for the execution. Some examples of such files are at scripts/configurations/base

Then, pass the configuration file to the script scripts/run.py

For each test, the directory structure is the following:

```
test_folder
├── test_name
│   ├── dataset_name
│   │   ├── embedding
│   │   │   ├── index1.index
│   │   │   └── ...
│   │   ├── lshforest
│   │   │   ├── forest_index1.json
│   │   │   └── ...
│   │   ├── results
│   │   │   ├── base
│   │   │   │   └── kK1_qQ1
│   │   │   │       ├── method1_results.csv
│   │   │   │       └── ...
│   │   │   ├── extracted
│   │   │   │   └── final_results_kK1_qQ1.csv
│   │   │   └── analyses
│   │   │       └── kK1_qQ1
│   │   │           ├── graph_precision.png
│   │   │           ├── false_positives.csv
│   │   │           └── ...
│   │   ├── statistics
│   │   │   ├── runtime.csv
│   │   │   ├── db.csv
│   │   │   └── storage.csv
│   │   ├── logging.log
│   │   ├── query_qQ1.json
│   │   └── query_qQ2.json
│   └── another_dataset
│       └── ...
└── another_test
    └── ...
```

In the main directory there will appear the log and query files.

In the "embedding" and "lshforest" folders there will be the structures used by those methods to complete the tests.
