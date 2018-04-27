#!/bin/sh
#

# Original working directory.
orig_dir=$(pwd)

# Directory where this script lives.
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -ne 1 ]; then
	echo "Project directory name must be provided."
else
	# Get the project name.
	proj_name=$(basename -- "$1")

	# Make the project directory and change to that directory.
	mkdir $1
	cd $1
	echo "C++ Project directory is being setup at: $orig_dir/$1"

	# Create the default directories.
	mkdir build doc include lib src test

	# Generate the default CMakeLists file.
	# Not sure why SED is creating a new file with -e appended.
	cp $script_dir/CMakeListsDefault* ./CMakeLists.txt
	sed -i -e "s/DEFAULT_PROJ_NAME/$proj_name/g" ./CMakeLists.txt
	rm CMakeLists.txt-e

	# Generate the default Doxyfile.in file.
	# Not sure why SED is creating a new file with -e appended.
	cd doc
	cp $script_dir/DoxyfileDefault* ./Doxyfile.in
	sed -i -e "s/DEFAULT_PROJ_NAME/$proj_name/g" ./Doxyfile.in
	rm Doxyfile.in-e

	# Change back to the original working directory.
	cd $orig_dir
fi

