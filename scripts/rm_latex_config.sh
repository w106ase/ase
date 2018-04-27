#!/bin/sh
#

# Define the filename for the LaTeX config file.
LTXCFGFILE=".ltxcfg"

# Search for an existing LaTeX config file.
orig_dir=$(pwd)
ltxcfg_file=$(find . -maxdepth 1 -name $LTXCFGFILE)

if [ -f "$ltxcfg_file" ]; then
  # Read the existing LaTeX config file.
  file=$(cat $ltxcfg_file)

  # Change to the main directory.
  cd $(dirname $file)
else
  # Find LaTeX file which contains \begin{document} statement.
  found_main=false
  for i in `seq 1 2`;
  do
    for file in *.tex; do
      if grep -q \begin{document} "$file"; then
        found_main=true
        break
      fi
    done

    if [ "$found_main" = true ]; then
      break
    else
      cd ..
    fi
  done
fi

# Find and remove all LaTeX config files below the main directory.
find . -name $LTXCFGFILE -exec rm -rf {} \;

# Change back to the original directory.
cd $orig_dir
