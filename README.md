# Setup

Scaricare la repository
```
git clone https://github.com/nanni00/tesi-magistrale.git /path/to/tesi-magistrale
```

Creare un ambiente python dedicato (tutto è stato provato su python 3.10, con altre versioni ci potrebbero essere problemi) con i pacchetti necessari (file environment.yml)

Aggiungere le variabili d'ambiente:
```
export THESIS_PATH=/path/to/tesi-magistrale
export PYTHONPATH=$PYTHONPATH:$THESIS_PATH
```

e per l'ambiente Go:
```
export GOPATH=$THESIS_PATH/go
export GOBIN=$GOPATH/bin
export PATH=$PATH:$GOBIN
```

I test e le azioni preliminari si possono eseguire direttamente sulle collection principali o sulle toy-collection indicando il parametro --size.
Prima di eseguire i test, occorre preparare il database MongoDB e le relative collection, aggiungendo un identificatore numerico a ogni documento e un vettore booleano che indichi le colonne numeriche.

Per aggiungere l'identificatore numerico,
```
python create_numeric_index_on_collections.py --task set
```
questa azione è reversibile (--task unset).


Per creare i metadati relativi alle colonne numeriche,
```
python detect_numeric_columns.py --task set --mode naive
```
anche qui ci sono alcune opzioni accessibili (e.g. nella modalità spacy le colonne vengono selezionate con un modo un po' più fine, ma forse non è così necessario e inoltre è computazionalmente più pesante). Anche questa azione è reversibile (--task unset).

L'esecuzione dei due passaggi è riassunta di fatto nello script scripts/sh/collection_preparation.sh.


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