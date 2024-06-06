#!/bin/bash

SET_TEST_NAME=set_small
BAG_TEST_NAME=bag_small


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --test-name $SET_TEST_NAME \
    --mode set \
    --tasks createindex \
    --dbname nanni \
    -k 10 \
    --small 
