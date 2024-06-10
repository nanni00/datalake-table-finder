#!/bin/bash

TEST_NAME=full

PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/results_basic_extraction.py
ALGORITHM=josie
MODE=bag
K=10
NUM_QUERY_SAMPLES=10

# LSHForest parameters
L=16
NUM_PERM=256

TEST=1
ANALYSE=1
CLEAN=0



if [[ $TEST -eq 1 ]]; then
    echo "######################### TESTING ##################################"
    python $PY_TESTER \
        --test-name $TEST_NAME \
        --algorithm $ALGORITHM \
        --mode $MODE \
        --tasks data-preparation query \
        --dbname nanni \
        --num-query-samples $NUM_QUERY_SAMPLES \
        -k $K \
        -l $L \
        --num-perm $NUM_PERM \
        --smal
fi


if [[ $ANALYSE -eq 1 ]]; then
    echo "######################### ANALYSIS ##################################"
    python $PY_RESULTS_ANALYSIS \
        --test-name $TEST_NAME \
        --algorithm $ALGORITHM \
        --mode $MODE \
        -k $K \
        --analyse-up-to 10 \
        --small 
fi


if [[ $CLEAN -eq 1 ]]; then
    echo "######################### CLEANING ##################################"
    python $PY_TESTER \
        --test-name $TEST_NAME \
        --dbname nanni \
        --small \
        --clean
fi