#!/bin/bash

TEST_NAME=main_test

# python scripts
PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/extract_results.py

# query generic parameters
K=10
NUM_QUERY_SAMPLES=1000

# JOSIE parameter
DBNAME=nanni

# LSHForest parameters
NUM_PERM=256
L=32

# Neo4j graph parameters
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678


# tasks
DATA_PREPRATION=1
SAMPLE_QUERIES=1
QUERY=1

ANALYSE=1
CLEAN=0

# used for tasks, in order to have the same queries for all the algorithms and modes
i=0

for ALGORITHM in josie lshforest
do
    for MODE in set bag
    do
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


        if [[ $CLEAN -eq 1 ]]; then
            echo "######################### CLEANING $ALGORITHM $MODE ##################################"
            python $PY_TESTER \
                --test-name $TEST_NAME \
                --algorithm $ALGORITHM \
                --mode $MODE \
                --dbname $DBNAME \
                --clean
        fi
    done
done




if [[ $ANALYSE -eq 1 ]]; then
    echo "######################### ANALYSIS ##################################"
    python $PY_RESULTS_ANALYSIS \
        --test-name $TEST_NAME
fi