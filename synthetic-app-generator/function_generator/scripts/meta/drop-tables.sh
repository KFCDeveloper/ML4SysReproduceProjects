for function in `ls build`; do echo $function; node /home/ubuntu/mongodb-scripts/drop-table.js $function; done
