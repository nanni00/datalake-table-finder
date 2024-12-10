"""
Simple script to start the Postgres and MongoDB servers
For Postgres, default location of the database files is at data/pg_data,
and here there's also the postgresql.conf file, where you can set the port
number, the WAL size...
"""

import os
import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=["start", "stop"])
    parser.add_argument('dbms', nargs='+', choices=['mongodb', 'pg'])
    parser.add_argument('--mongodb-local', required=False, action="store_true")
    parser.add_argument('--pg-path', required=False)
    parser.add_argument('--mongo-root-path', required=False)
    parser.add_argument('--pg-config-path', required=False)

    args = parser.parse_args()
    dthpath = os.environ['DLTFPATH']
    home = os.environ['HOME']

    task = args.task
    mongo_path =    f"{home}/mongodb-srv"           if not args.mongo_root_path else args.mongo_root_path
    pg_path =       f"{dthpath}/data/pg_data"       if not args.pg_path else args.pg_path
    pg_conf_path =  f"{pg_path}/postgresql.conf"    if not args.pg_config_path else args.pg_config_path

    for dbanme in args.dbms:
        match dbanme:
            case 'pg':
                match task:
                    case 'start':
                        subprocess.run(f"pg_ctl start -D {pg_path} -o \"-c config_file={pg_conf_path}\"", shell=True)
                    case 'stop':
                        subprocess.run(f"pg_ctl stop -D {pg_path} -o \"-c config_file={pg_conf_path}\"", shell=True)
            
            case 'mongodb':
                match task:
                    case 'start':
                        if args.mongodb_local:
                            subprocess.call(f"\"{mongo_path}/bin/mongod\" --dbpath \"{mongo_path}/data/\" --logpath \"{mongo_path}/log/mongod.log\" --fork", shell=True)
                        else:
                            subprocess.call("sudo service mongod start", shell=True)
                    case 'stop':
                        if args.mongodb_local:
                            subprocess.call(f"\"{mongo_path}/bin/mongod\" --shutdown --dbpath \"{mongo_path}/data/\" --logpath \"{mongo_path}/log/mongod.log\"", shell=True)
                        else:
                            subprocess.call("sudo service mongod stop", shell=True)
        