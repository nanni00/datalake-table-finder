#!/bin/bash


python $THESIS_PATH/experiments/josie/josie_testing.py --test-name set1k_jar --mode set --dbname nanni --tasks all -k 10 --tables-limit 1000 --use-scala
