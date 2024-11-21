# Data Lake Table Finder

This repository contains scripts and files to compare different methods in data discovery tasks.

# Setup

1. Download the repository

2. Create a python environment (see environment.yml)

3. Create the project environment variable and add it to the PYTHONPATH
```
export DLTF=/path/to/datalake-table-finder
export PYTHONPATH=$PYTHONPATH:$THESIS_PATH
```

4. Load the datalake on MongoDB, or store it as a local file (e.g. SantosLarge)

I datalake usati nei test possono essere o su un database MongoDB oppure si possono strutturare come una cartella contenente dei CSV, ogniuno dei quali è una tabella del corpus.

Prima di eseguire i test, occorre svolgere una fase di preparazione per costruire un mapping tra l'ID della tabella (su MongoDB il campo '_id', altrimenti il nome del file CSV) e un ID numerico; inoltre bisogna determinare quali colonne sono numeriche.

Su MongoDB, per aggiungere l'identificatore numerico,
```
python create_numeric_index_on_collections.py --task set
```

e per determinare le colonne numeriche
```
python detect_numeric_columns.py --task set --mode naive
```
L'esecuzione dei due passaggi è riassunta di fatto nello script scripts/sh/collection_preparation.sh.

Se si lavora con una cartella di file CSV, lo script scripts/py/preproc/datalake_csv_preparation.py.


# Esecuzione dei test

Per eseguire i test, il modo principale consiste nel definire un file JSON di configurazione (alcuni esempi sono in scripts/py/configurations) e passarlo come parametro a scripts/py/parser_test_configuration.py:
```
python parser_test_configuration.py /path/to/configuration.json
``` 

Nel file di configurazione è possibile specificare tutti i parametri necessari ai vari step e quali di questi eseguire (e.g. solo la data preparation, le analisi finali, etc).

Per ogni test la struttura degli elementi prodotti è la seguente:
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

Definito il nome del test e il dataset su cui si lavora, nella cartella principale si trovano il file di log e i file JSON contenenti gli ID delle tabelle usate in fase di query (query_qQ1.json). 

Nelle cartelle "embedding" e "lshforest" ci sono le strutture di supporto per i relativi metodi, con indicato nel nome del file la modalità di riferimento (e.g. 'set', 'bag', etc).

Nella cartella "statistics", il file "runtime.csv" dà informazioni sulle tempistiche delle varie parti, in secondi, il file "storage.csv" sullo spazio su disco occupato, in GB, e il file "db.csv" dà informazioni più in dettaglio per le tabelle usate da JOSIE nel database PostgreSQL.

Nella cartella "results" ci sono tre sottoparti: in "base" ci sono i risultati grezzi forniti dai vari metodi, suddivisi per dimensione del query benchmark e per il valore di K; in "extracted" questi sono raggruppati in un unico file (sempre in relazione a Q e K) e vengono aggiunte altre informazioni, quali SLOTH overlap, Jaccard similarity etc; in "analyses" ci sono le analisi conclusive, con i vari grafici e tabelle.