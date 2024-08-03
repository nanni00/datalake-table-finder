#!/bin/bash

# start MongoDB - admin privileges required
# service mongod start

# start PostgreSQL
pg_ctl start -D $THESIS_PATH/data/pg_data -o "-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf"

