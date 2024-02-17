set -e

for function in `cat collected-functions.txt`
do
	# echo $function;
	substr="_cpu"
	prefix=${function%%$substr*}
	index=${#prefix}
	name=${function:0:index}
	echo $name >> function-names.txt
done

cat function-names.txt | sort | uniq > function-names-uniq.txt

for function in `head -n -1 function-names-uniq.txt`
do
	if grep -Fxq "$function" dumped-tables.txt
	then
		continue
	else
		# code if not found
		echo $function;
		node ./mongodb-scripts/dump-table.js $function
		echo $function >> dumped-tables.txt
	fi
done
