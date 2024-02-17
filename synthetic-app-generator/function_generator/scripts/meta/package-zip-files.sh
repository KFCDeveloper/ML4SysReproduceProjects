# Usage: ./package-zip-files.sh
cd build

for function in `ls`; do echo $function; cd $function; for dir in `ls`; do echo $dir; cd $dir; npm install --production; zip -rq $dir.zip *; cd ..; done; cd ..; done
