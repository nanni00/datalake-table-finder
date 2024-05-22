#!/usr/bin/bash

# stop MongoDB 
service mongod stop

# stop PostgreSQL
pg_ctl stop -D $THESIS_PATH/data/josie-tests/pg_data -o "-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf"

