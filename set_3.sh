#!/bin/bash
echo "Run Handin assignment 3"

echo "Creating the plotting and output directories"

if [ ! -d "plots" ]; then
  echo "Directory for plots does not exist create it!"
  mkdir plots
fi

if [ ! -d "output" ]; then
  echo "Directory for output does not exist create it!"
  mkdir output
fi

echo "Running minimization script"
python3 minimize.py > output/minimize.txt

echo "Running chisquare script"
python3 chisquare.py > output/chisquare.txt

echo "Running likelihood script"
python3 likelihood.py > output/likelihood.txt

echo "Running stat_test script"
python3 stat_test.py > output/stat_test.txt

echo "Generating the pdf"

pdflatex TeX/report_3.tex 
bibtex report_3.aux 
pdflatex TeX/report_3.tex 
pdflatex TeX/report_3.tex 

# remove all the extra files
rm -f *.aux
rm -f *.log
rm -f *.bbl
rm -f *.blg
rm -f *.out
rm -f *.toc
rm -f *.gz
rm -f *.fls
rm -f *.fdb_latexmk
rm -f *.synctex.gz

echo "Done"