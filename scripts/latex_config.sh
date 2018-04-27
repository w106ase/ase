#!/bin/sh
#

# Define function for finding the relative path between two absolute paths.
relpath(){ python -c "import os.path; print os.path.relpath('$1','${2:-$PWD}')" ; }

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

  # Get the filename with no path or extension.
  file=$(basename $file)
  file=$(echo $file | cut -f 1 -d '.' )
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

  # Remove the extension.
  file=$(echo $file | cut -f 1 -d '.')

  # Relative path to the main file from original directory.
  rel_path="$(relpath $(pwd) $orig_dir)/" 

  # Write to file.
  echo "$rel_path$file" >> "$orig_dir/$LTXCFGFILE"
fi

makeglossaries $file
pdflatex --synctex=1 $file
bibtex $file

makeglossaries $file
pdflatex --synctex=1 $file
bibtex $file

makeglossaries $file
pdflatex --synctex=1 $file
bibtex $file

open -a Skim -g "$file.pdf"
rm -f *.{acn,acr,alg,aux,bbl,blg,fdb_latexmk,fls,glg,glsdefs,ist,log,out,sbl,sym}

# Change back to the original directory.
cd $orig_dir

