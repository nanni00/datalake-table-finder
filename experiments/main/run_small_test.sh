#!/bin/bash

TEST_NAME=set_jdbc

PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/compare_josie_and_sloth.py


python $PY_TESTER \
    --test-name $TEST_NAME \
    --mode set \
    --tasks j-createindex samplequeries j-dbsetup j-query \
    --dbname nanni \
    -k 10 \
    --small

python $PY_RESULTS_ANALYSIS \
    --test-name $TEST_NAME \
    -k 10 \
    --analyse-up-to 10 \
    --small \

# python $PY_TESTER \
#     --test-name $TEST_NAME \
#     --dbname nanni \
#     --small \
#     --clean
