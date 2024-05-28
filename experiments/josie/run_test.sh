#!/bin/bash

SET_TEST_NAME=diffhash_set_2m
BAG_TEST_NAME=diffhash_bag_2m


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --dbname nanni \
    --tables-limit 2000000 \
    --mode set \
    --tasks all \
    -k 10 \
    --test-name $SET_TEST_NAME


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --test-name $BAG_TEST_NAME \
    --mode bag \
    --tasks createindex dbsetup josietest \
    --dbname nanni \
    --tables-limit 2000000 \
    --queries-file $THESIS_PATH/data/josie-tests/$SET_TEST_NAME/queries.json \
    --convert-query-ids \
    -k 10  

