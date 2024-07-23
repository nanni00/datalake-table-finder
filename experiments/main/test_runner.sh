#!/bin/bash

TEST_NAME=test_lshforest

# python scripts
PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_EXTRACTION=$THESIS_PATH/experiments/main/extract_results.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/analysis.py

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

NUM_CPU=72

# ALGORITHMS="josie lshforest embedding"
# MODES="set bag fasttext"

ALGORITHMS="lshforest embedding"
MODES="set bag fasttext"

# tasks
DATA_PREPRATION=1
SAMPLE_QUERIES=0
QUERY=1

# query sizes in term of number of queries
QUERY_SIZES="1000"
# QUERY_SIZES="1000 10000 100000"

# extract more information from initial results (like SLOTH overlap for each table pair)
EXTRACT=0

# do the concrete analyses
ANALYSE=0

# remove database tables and big files
CLEAN=0

for ALGORITHM in $ALGORITHMS
do
    for MODE in $MODES
    do
        i=0
        TASKS=''

        # if there are multiple query size (e.g. 10, 100, 1000...)
        # do the data preparation step only in the first run
        if [[ $i -eq 0 && $DATA_PREPRATION -eq 1 ]]; then
            TASKS="data-preparation"
        fi

        i=$((i+1))

        if [[ $SAMPLE_QUERIES -eq 1 ]]; then
            TASKS="${TASKS} sample-queries"
        fi

        if [[ $QUERY -eq 1 ]]; then
            TASKS="${TASKS} query"
        fi

        # run the program with all the parameters
        if [[ $DATA_PREPRATION -eq 1 || $SAMPLE_QUERIES -eq 1 || $QUERY -eq 1 ]]; then
            echo "######################### TESTING $ALGORITHM $MODE TASK=$TASKS K=$K ##################################"
            python $PY_TESTER \
                --test-name $TEST_NAME \
                --algorithm $ALGORITHM \
                --mode $MODE \
                --tasks $TASKS \
                --dbname $DBNAME \
                --num-query-samples $QUERY_SIZES \
                -k $K \
                -l $L \
                --neo4j-user $NEO4J_USER \
                --neo4j-password $NEO4J_PASSWORD \
                --num-perm $NUM_PERM \
                --num-cpu $NUM_CPU \
                --small
        fi
    done
done


if [[ $CLEAN -eq 1 ]]; then
    for ALGORITHM in $ALGORITHMS
    do
        for MODE in $MODES
        do
            echo "######################### CLEANING $ALGORITHM $MODE ##################################"
            python $PY_TESTER \
                --test-name $TEST_NAME \
                --algorithm $ALGORITHM \
                --mode $MODE \
                --dbname $DBNAME \
                --clean
        done
    done
fi


if [[ $EXTRACT -eq 1 ]]; then
    for NUM_QUERY_SAMPLES in $QUERY_SIZES
    do        
        echo "######################### EXTRACTION $NUM_QUERY_SAMPLES ##################################"
        python $PY_RESULTS_EXTRACTION \
            --test-name $TEST_NAME \
            --num-query-samples $NUM_QUERY_SAMPLES \
            --num-cpu $NUM_CPU \
            --dbname $DBNAME
    done
fi



if [[ $ANALYSE -eq 1 ]]; then
    for NUM_QUERY_SAMPLES in $QUERY_SIZES
    do        
        echo "######################### ANALYSIS $NUM_QUERY_SAMPLES ##################################"
        python $PY_RESULTS_ANALYSIS \
            --test-name $TEST_NAME \
            --num-query-samples $NUM_QUERY_SAMPLES \
            --num-cpu $NUM_CPU
    done
fi
