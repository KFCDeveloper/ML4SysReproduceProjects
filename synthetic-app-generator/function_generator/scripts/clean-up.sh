# Usage: ./clean-up.sh
cd ../build

for function in `ls`; do echo $function; cd $function; for dir in `ls`; do echo $dir; cd $dir; rm -rf node_modules package-lock.json; cd ..; done; cd ..; done