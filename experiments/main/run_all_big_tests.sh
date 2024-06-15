#!/bin/bash

TEST_NAME=bigtest

# python scripts
PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/results_basic_extraction.py

# query generic parameters
K=10
NUM_QUERY_SAMPLES=500

# JOSIE parameter
DBNAME=nanni

# LSHForest parameters
L=16
NUM_PERM=256

# tasks
DATA_PREPRATION=0
SAMPLE_QUERIES=0
QUERY=0

ANALYSE=0
CLEAN=1

# used for tasks, in order to have the same queries for all the algorithms and modes
i=0

for ALGORITHM in josie lshforest embedding
do
    for MODE in set bag fasttext
    do
        # check is the configuration is correct or not
        if [[ $ALGORITHM != "embedding" && $MODE == "fasttext" ]]; then
            continue
        fi
        
        if [[ $ALGORITHM == "embedding" && $MODE != "fasttext" ]]; then
            continue
        fi


        TASKS=''

        if [[ $DATA_PREPRATION -eq 1 ]]; then
            TASKS="data-preparation"
        fi

        if [[ $i -eq 0 && $SAMPLE_QUERIES -eq 1 ]]; then
            TASKS="${TASKS} sample-queries"
        fi

        if [[ $QUERY -eq 1 ]]; then
            TASKS="${TASKS} query"
        fi

        i=$(( i + 1 ))

        if [[ $DATA_PREPRATION -eq 1 || $SAMPLE_QUERIES -eq 1 || $QUERY -eq 1 ]]; then
            echo "######################### TESTING $ALGORITHM $MODE $TASKS $K ##################################"
            python $PY_TESTER \
                --test-name $TEST_NAME \
                --algorithm $ALGORITHM \
                --mode $MODE \
                --tasks $TASKS \
                --dbname $DBNAME \
                --num-query-samples $NUM_QUERY_SAMPLES \
                -k $K \
                -l $L \
                --num-perm $NUM_PERM
        fi


        if [[ $ANALYSE -eq 1 ]]; then
            echo "######################### ANALYSIS $ALGORITHM $MODE ##################################"
            python $PY_RESULTS_ANALYSIS \
                --test-name $TEST_NAME \
                --algorithm $ALGORITHM \
                --mode $MODE \
                -k $K \
                --analyse-up-to 10
        fi


        if [[ $CLEAN -eq 1 ]]; then
            echo "######################### CLEANING $ALGORITHM $MODE ##################################"
            python $PY_TESTER \
                --test-name $TEST_NAME \
                --dbname $DBNAME \
                --clean
        fi
    done
done
