#!/bin/bash

TEST_NAME=full

PY_TESTER=$THESIS_PATH/experiments/main/main_tester.py
PY_RESULTS_ANALYSIS=$THESIS_PATH/experiments/main/compare_josie_and_sloth.py
ALGORITHM=lshforest
MODE=bag
K=10

python $PY_TESTER \
    --test-name $TEST_NAME \
    --algorithm $ALGORITHM \
    --mode $MODE \
    --tasks query \
    --dbname nanni \
    -k $K \
    --small


# python $PY_RESULTS_ANALYSIS \
#     --test-name $TEST_NAME \
#     --mode $MODE \
#     -k $K \
#     --analyse-up-to 10 \
#     --small \


# python $PY_TESTER \
#     --test-name $TEST_NAME \
#     --dbname nanni \
#     --small \
#     --clean
