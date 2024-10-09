import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=["start", "stop"])
    parser.add_argument('--mongodb-local', required=False, action="store_true")
    parser.add_argument('databases', nargs='+', choices=['mongodb', 'pg'])
    parser.add_argument('--pg-path', required=False)
    parser.add_argument('--mongo-root-path', required=False)

    args = parser.parse_args()

    pg_path = f"$THESIS_PATH/data/pg_data" if not args.pg_path else args.pg_path
    mongo_path = f"$HOME/mongodb-srv" if not args.mongo_root_path else args.mongo_root_path

    task = args.task

    for dbanme in args.databases:
        match dbanme:
            case 'pg':
                match task:
                    case 'start':
                        subprocess.run(f"pg_ctl start -D {pg_path} -o \"-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf\"", shell=True)
                    case 'stop':
                        subprocess.run(f"pg_ctl stop -D {pg_path} -o \"-c config_file=$THESIS_PATH/go/src/github.com/ekzhu/josie/conf/postgresql.conf\"", shell=True)
            
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
        