# Set up

Scaricare la repository
```
git clone https://github.com/nanni00/tesi-magistrale.git /path/to/tesi-magistrale
```

Creare un ambiente python dedicato (tutto è stato provato su python 3.10, con altre versioni ci potrebbero essere problemi) e scaricare i pacchetti necessari
```
pip install -r requirements.txt
```

Aggiungere le variabili d'ambiente generiche:
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

I dati prodotti dal test complessivo sono nella cartella /data/josie-tests/\<nome-test\>, dove \<nome-test\> è un nome definito dall'utente oppure un codice del tipo m\<modalità\> in cui "modalità" è "set" se si lavora con la set semantic, o "bag" se si lavora con la bag semantic.

Per fare i test con JOSIE è anche necessario:
- Scaricare il dataset originale usato per il training di TURL (https://github.com/sunlab-osu/TURL.git) (train_tables.jsonl nella cartella OneDrive) e posizionarlo in data/turl_sloth/wikitables/original_turl_train_tables.jsonl o, in alternativa, avere un database MongoDB con già caricato sopra queste tabelle nella collection "optitab.wikitables";
- scaricare PostgreSQL (seguire le istruzioni sulla repository https://github.com/ekzhu/josie.git);
- creare un database con lo stesso nome \<nome-test\> in cui verranno caricate le tabelle usate da JOSIE;
- è possibile che vadano scaricati i pacchetti necessari per il codice Go, indicati nel .mod se non già presenti;

Per eseguire il test c'è lo script python experiments/josie/josie_testing.py che si può configurare con i vari argomenti.
