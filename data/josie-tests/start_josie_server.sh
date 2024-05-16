#!/usr/bin/bash

postgres -D $THESIS_PATH/data/josie-tests/pg_data -c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf

