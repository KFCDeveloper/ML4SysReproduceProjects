cd build
for function in `ls`; do echo $function; cd $function; for dir in `ls`; do echo $dir; cd $dir; mem="`echo $dir | rev | cut -d'_' -f 1 | rev`"; cpu="`echo $dir | rev | cut -d'_' -f 3 | rev`"; echo "cpu=$cpu mem=$mem"; wsk -i action invoke "$dir-$cpu" -r --blocking; sleep 1; cd ..; break; done; cd ..; done

cd ..
