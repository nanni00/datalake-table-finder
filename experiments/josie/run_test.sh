#!/bin/bash

TEST_NAME=settest

# create the inverted index and the integer sets files, ready to be uploaded to the PostgreSQL database (which may not be running yet)
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $TEST_NAME --mode set -k 5 --tasks createindex --dbname nanni

# sample the query IDs that will be used in the next steps
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $TEST_NAME --mode set -k 5 --tasks samplequeries --dbname nanni

# load into the PostgreSQL the inverted index, the integer sets and the queries
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $TEST_NAME --mode set -k 5 --tasks dbsetup --dbname nanni

# run the JOSIE tests
python $THESIS_PATH/experiments/josie/josie_testing.py --test-name $TEST_NAME --mode set -k 5 --tasks josietest --dbname nanni

