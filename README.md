Scaricare la repository
```
git clone https://github.com/nanni00/tesi-magistrale.git /path/to/tesi-magistrale
```

Creare un ambiente python dedicato (tutto è stato provato su python 3.10, con altre versioni ci potrebbero essere problemi) e scaricare i pacchetti necessari (file requirements.txt o environment.yml)

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

I test e le azioni preliminari si possono eseguire direttamente sulle collection principali o sulle toy-collection indicando il parametro --small.
Prima di eseguire i test, occorre preparare il database MongoDB e le relative collection, aggiungendo un identificatore numerico a ogni documento e un vettore booleano che indichi le colonne numeriche.

Per aggiungere l'identificatore numerico,

```
python create_numeric_index_on_collections.py --task set
```

questa azione è reversibile (vd --task unset).


Per creare i vettori relativi alle colonne numeriche,

```
python detect_numeric_columns.py --task set --mode naive
```

anche qui ci sono alcune opzioni accessibili (e.g. nella modalità spacy le colonne vengono selezionate con un modo un po' più fine, ma forse non è così necessario e inoltre è computazionalmente più pesante). Anche questa azione è reversibile (vd --task unset).

Una volta completato il set up, si possono eseguire i test,

```
./run_all_<small/big>_tests.sh
```

indicando quali parti eseguire tra testing, analisi e cleaning (la parte di cleaning non rimuove quanto fatto nel setup indicato sopra).




