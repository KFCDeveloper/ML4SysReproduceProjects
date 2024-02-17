set -e

cd build

for function in `ls`;
do
	echo $function; cd $function;
	for dir in `ls`;
	do
		echo $dir; cd $dir;
		# Create the action package
		if ls ./*.zip 1> /dev/null 2>&1; then
			echo "ZIP file exist!"
		else
			echo "ZIP file does not exist!"
			npm install --production
			zip -rq $dir.zip .
		fi

		# Add the action to OWK
		echo "Adding the action to OpenWhisk..."
		mem="`echo $dir | rev | cut -d'_' -f 1 | rev`";
		cpu="`echo $dir | rev | cut -d'_' -f 3 | rev`";
		echo "cpu=$cpu mem=$mem";
		wsk -i action create "$dir-$cpu" -m "$mem" --kind nodejs:14 "$dir.zip";

		# Run the functions
		echo "Running the function..."
		# for i in {1..20}
		# do
		# 	wsk -i action invoke "$dir-$cpu";
		# done
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
		wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
                wsk -i action invoke "$dir-$cpu";
		sleep 25

		# Delete the action from OWK
		wsk -i action delete "$dir-$cpu";
		echo "Action deleted!"

		cd ..;

		# Delete the directory
		rm -rf $dir;
		echo "Directory deleted!"
		echo "$dir" >> "../../collected-functions.txt"
		# exit 0
	done;
	cd ..;
done

cd ..
