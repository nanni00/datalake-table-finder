#!/bin/bash


# stop MongoDB - admin privileges required
# service mongod stop

# stop MongoDB - with Tarball installation
# "$HOME/mongodb-srv/bin/mongod" --shutdown --dbpath "$HOME/mongodb-srv/data/" --logpath "$HOME/mongodb-srv/log/mongod.log" 

# stop PostgreSQL
pg_ctl stop -D $THESIS_PATH/data/pg_data -o "-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf"

