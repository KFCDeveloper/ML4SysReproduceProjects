# wsk action create $dir --kind nodejs:14 $dir.zip
cd build
for function in `ls`; do echo $function; cd $function; for dir in `ls`; do echo $dir; cd $dir; mem="`echo $dir | rev | cut -d'_' -f 1 | rev`"; cpu="`echo $dir | rev | cut -d'_' -f 3 | rev`"; echo "cpu=$cpu mem=$mem"; wsk -i action create "$dir-$cpu" -m "$mem" --kind nodejs:14 "$dir.zip"; cd ..; break; done; cd ..; done

cd ..