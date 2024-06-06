#!/bin/bash

SET_TEST_NAME=set2m
BAG_TEST_NAME=bag2m


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --test-name $SET_TEST_NAME \
    --mode set \
    --tasks all \
    --dbname nanni \
    --tables-limit 2000000 \
    -k 10


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --test-name $BAG_TEST_NAME \
    --mode bag \
    --tasks all \
    --dbname nanni \
    --tables-limit 2000000 \
    -k 10  

