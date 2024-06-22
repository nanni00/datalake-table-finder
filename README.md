Scaricare la repository
```
git clone https://github.com/nanni00/tesi-magistrale.git /path/to/tesi-magistrale
```

Creare un ambiente python dedicato (tutto è stato provato su python 3.10, con altre versioni ci potrebbero essere problemi) e scaricare i pacchetti necessari (file environment.yml)

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

questa azione è reversibile (--task unset).


Per creare i metadati relativi alle colonne numeriche,

```
python detect_numeric_columns.py --task set --mode naive
```

anche qui ci sono alcune opzioni accessibili (e.g. nella modalità spacy le colonne vengono selezionate con un modo un po' più fine, ma forse non è così necessario e inoltre è computazionalmente più pesante). Anche questa azione è reversibile (--task unset).

Per eseguire i test c'è lo script experiments/main/test_runner.sh, che può essere configurato in vari modi.

I risultati iniziali vengono raccolti nella cartella data/tests/\<test-name\>/results/base; una volta che questi vengono analizzati (cioè viene calcolato l'overlap reale con SLOTH e si raccolgono altre informazioni) i risultati finali vengono portati nel file data/tests/\<test-name\>/results/extracted. Per eseguire le analisi c'è il parametro ANALYSE nello script test_runner.sh.

Nel notebook analysis.ipynb ci sono i passaggi per fare alcune operazioni, come il calcolo della precision@p e del ndcg@p, oltre che per valutare quanto ci impiegano i vari metodi come query time per singola query.


Tips: per JOSIE, se è già stato



