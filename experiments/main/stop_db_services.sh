#!/bin/bash

# stop MongoDB - admin privileges required
# service mongod stop

# stop PostgreSQL
pg_ctl stop -D $THESIS_PATH/data/tests/pg_data -o "-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf"

