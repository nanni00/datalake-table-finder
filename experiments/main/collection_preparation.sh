#!/bin/bash

# Python scripts
PY_ID_NUMERIC_SCRIPT=$THESIS_PATH/experiments/main/create_numeric_index_on_collections.py
PY_DETECT_NUMERIC_COLUMNS_SCRIPT=$THESIS_PATH/experiments/main/detect_numeric_columns.py

TASK="set"
DATASET="gittables"
SMALL=0

# only for numeric columns detection
NUMERIC_MODE_DETECTION="naive"
NUM_CPU=72


# adding the '_id_numeric' field
if [[ $SMALL -eq 0 ]] 
then
    python $PY_ID_NUMERIC_SCRIPT --task $TASK --dataset $DATASET
else
    python $PY_ID_NUMERIC_SCRIPT --task $TASK --dataset $DATASET --small
fi

# detecting numeric columns
if [[ $SMALL -eq 0 ]] 
then
    python $PY_DETECT_NUMERIC_COLUMNS_SCRIPT --task $TASK --mode $MODE --num-cpu $NUM_CPU --dataset $DATASET
else
    python $PY_DETECT_NUMERIC_COLUMNS_SCRIPT --task $TASK --mode $MODE --num-cpu $NUM_CPU --dataset $DATASET --small
fi

