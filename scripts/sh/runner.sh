#!/bin/bash

TEST_NAME=a_test

MEMORY_PROFILING=0
MEMORY_REPORTS_DIR=$THESIS_PATH/scripts/py/memory_reports

PY_SCRIPTS_PATH=$THESIS_PATH/scripts/py

# python scripts
PY_TESTER=$PY_SCRIPTS_PATH/main_pipeline.py
PY_RESULTS_EXTRACTION=$PY_SCRIPTS_PATH/extract_results.py
PY_RESULTS_ANALYSIS=$PY_SCRIPTS_PATH/analysis_pl.py


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

# number of cores used in parallel tasks
NUM_CPU=72

# ALGORITHMS="josie"
# MODES="bag"
ALGORITHMS="josie lshforest embedding"
MODES="set bag fasttext fasttextdist"

# dataset
DATASETS="wikipedia"
SIZE="standard"

# query sizes in term of number of queries
QUERY_SIZES="50000"
# QUERY_SIZES="1000 10000 100000"

# values used for analyses
# P_VALUES="10 15 20"
P_VALUES="1 3 5 10"


# tasks
DATA_PREPRATION=0
SAMPLE_QUERIES=0
QUERY=0
EXTRACT=0   # extract more information from initial results (like SLOTH overlap for each table pair) 
ANALYSE=1   # do the concrete analyses
CLEAN=0   # remove database tables and big files

for DATASET in $DATASETS 
do
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

            
            if [[ $DATA_PREPRATION -eq 1 || $SAMPLE_QUERIES -eq 1 || $QUERY -eq 1 ]]; then
                if [[ $MEMORY_PROFILING -eq 0 ]]; then    
                # run the program with all the parameters
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
                        --dataset $DATASET \
                        --size $SIZE \
                        --token-table-on-memory
                else
                    REPORT_FILE="${MEMORY_REPORTS_DIR}/main_tester.${TEST_NAME}_${DATASET}_${ALGORITHM}_${MODE}.bin"
                    memray run --native --follow-fork --output $REPORT_FILE \
                        $PY_TESTER \
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
                        --dataset $DATASET \
                        --size $SIZE
                fi
            fi
        done
    done
    
    if [[ $CLEAN -eq 1 ]]; then
        for ALGORITHM in $ALGORITHMS
        do
            for MODE in $MODES
            do
                python $PY_TESTER \
                    --test-name $TEST_NAME \
                    --algorithm $ALGORITHM \
                    --mode $MODE \
                    --dbname $DBNAME \
                    --dataset $DATASET \
                    --size $SIZE \
                    --clean
            done
        done
    fi


    if [[ $EXTRACT -eq 1 ]]; then    
        for NUM_QUERY_SAMPLES in $QUERY_SIZES
        do
            if [[ $MEMORY_PROFILING -eq 0 ]]
            then
                python $PY_RESULTS_EXTRACTION \
                    --test-name $TEST_NAME \
                    --num-query-samples $NUM_QUERY_SAMPLES \
                    --num-cpu $NUM_CPU \
                    --dbname $DBNAME \
                    --dataset $DATASET \
                    -k $K \
                    --size $SIZE
            else
                REPORT_FILE="${MEMORY_REPORTS_DIR}/extraction.${TEST_NAME}_${NUM_QUERY_SAMPLES}.bin"
                memray run --native --follow-fork --output $REPORT_FILE \
                    $PY_RESULTS_EXTRACTION \
                    --test-name $TEST_NAME \
                    --num-query-samples $NUM_QUERY_SAMPLES \
                    --num-cpu $NUM_CPU \
                    --dbname $DBNAME \
                    --dataset $DATASET \
                    -k $K \
                    --size $SIZE

            fi
        done
    fi


    if [[ $ANALYSE -eq 1 ]]; then
        for NUM_QUERY_SAMPLES in $QUERY_SIZES
        do        
            python $PY_RESULTS_ANALYSIS \
                --test-name $TEST_NAME \
                --num-query-samples $NUM_QUERY_SAMPLES \
                --num-cpu $NUM_CPU \
                --dataset $DATASET \
                -k $K \
                --size $SIZE \
                --p-values $P_VALUES
        done
    fi
done