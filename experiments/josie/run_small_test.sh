#!/bin/bash

SET_TEST_NAME=set_small_str2


python $THESIS_PATH/experiments/josie/josie_testing.py \
    --test-name $SET_TEST_NAME \
    --mode set \
    --tasks updatequeries josietest \
    --dbname nanni \
    -k 10 \
    --small \
    --queries-file /data4/nanni/tesi-magistrale/data/josie-tests/set_small_str/queries.json
    # --tasks dbsetup \

BAG_TEST_NAME=bag_small_str