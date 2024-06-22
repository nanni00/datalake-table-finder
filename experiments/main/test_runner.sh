#!/bin/bash

TEST_NAME=main_test

# python scripts
PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/extract_results.py

# query generic parameters
K=10

# JOSIE parameter
DBNAME=nanni

# LSHForest parameters
NUM_PERM=256
L=32

# Neo4j graph parameters
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678


# tasks
DATA_PREPRATION=0
SAMPLE_QUERIES=0
QUERY=0

# extract more information from initial results (like SLOTH overlap for each table pair)
ANALYSE=1

# remove database tables and big files
CLEAN=0

# used for tasks, in order to have the same queries for all the algorithms and modes
i=0

for ALGORITHM in josie lshforest
do
    for MODE in set bag
    do
        for NUM_QUERY_SAMPLES in 10 # 1000 10000 100000
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
                    --neo4j-user $NEO4J_USER \
                    --neo4j-password $NEO4J_PASSWORD \
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
done



for NUM_QUERY_SAMPLES in 10 #1000 10000 100000
do        
    if [[ $ANALYSE -eq 1 ]]; then
        echo "######################### ANALYSIS $NUM_QUERY_SAMPLES ##################################"
        python $PY_RESULTS_ANALYSIS \
            --test-name $TEST_NAME \
            --num-query-samples $NUM_QUERY_SAMPLES
    fi
done