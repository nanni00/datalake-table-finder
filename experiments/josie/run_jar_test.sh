#!/bin/bash

PY_TEST_NAME=set1m_py
JAR_TEST_NAME=set1m_jar

# create the inverted index and the integer sets files, ready to be uploaded to the PostgreSQL database (which may not be running yet)
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $PY_TEST_NAME  --mode set -k 20 --tasks all --dbname nanni --tables-limit 1000000
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $JAR_TEST_NAME --mode set -k 20 --tasks all --dbname nanni --tables-limit 1000000 --use-scala
