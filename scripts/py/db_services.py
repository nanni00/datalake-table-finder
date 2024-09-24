#!/home/giovanni.malaguti/miniconda3/envs/nanni3.10/bin/python
import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=["start", "stop"])
    parser.add_argument('--mongodb-local', required=False, action="store_true")
    parser.add_argument('databases', nargs='+', choices=['mongodb', 'pg'])

    args = parser.parse_args()

    task = args.task
    for dbanme in args.databases:
        match dbanme:
            case 'pg':
                match task:
                    case 'start':
                        subprocess.run("pg_ctl start -D $THESIS_PATH/data/pg_data -o \"-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf\"", shell=True)
                    case 'stop':
                        subprocess.run("pg_ctl stop -D $THESIS_PATH/data/pg_data -o \"-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf\"", shell=True)
            case 'mongodb':
                match task:
                    case 'start':
                        if args.mongodb_local:
                            subprocess.call("\"$HOME/mongodb-srv/bin/mongod\" --dbpath \"$HOME/mongodb-srv/data/\" --logpath \"$HOME/mongodb-srv/log/mongod.log\" --fork", shell=True)
                        else:
                            subprocess.call("sudo service mongod start", shell=True)
                    case 'stop':
                        if args.mongodb_local:
                            subprocess.call("\"$HOME/mongodb-srv/bin/mongod\" --shutdown --dbpath \"$HOME/mongodb-srv/data/\" --logpath \"$HOME/mongodb-srv/log/mongod.log\"", shell=True)
                        else:
                            subprocess.call("sudo service mongod stop", shell=True)
        