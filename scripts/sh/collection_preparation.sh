#!/bin/bash

# Python scripts
PY_ID_NUMERIC_SCRIPT=$THESIS_PATH/scripts/py/preproc/create_numeric_index_on_collections.py
PY_DETECT_NUMERIC_COLUMNS_SCRIPT=$THESIS_PATH/scripts/py/preproc/detect_numeric_columns.py

TASK="set"
DATASET="wikiturlsnap"
SIZE="standard"

# only for numeric columns detection
NUMERIC_DETECTION_MODE="naive"
NUM_CPU=72


# adding the '_id_numeric' field
# echo "CREATING '_id_numeric' ID ON DATASET $DATASET"
# python $PY_ID_NUMERIC_SCRIPT --task $TASK --dataset $DATASET --size $SIZE


# detecting numeric columns
echo "DETECTIG NUMERIC COLUMNS ON DATASET $DATASET"
python $PY_DETECT_NUMERIC_COLUMNS_SCRIPT --task $TASK --mode $NUMERIC_DETECTION_MODE --num-cpu $NUM_CPU --dataset $DATASET --size $SIZE

