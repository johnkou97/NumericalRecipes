#!/bin/bash
echo "Run Handin assignment 1"

echo "Creating the plotting and output directories"

if [ ! -d "plots" ]; then
  echo "Directory for plots does not exist create it!"
  mkdir plots
fi

if [ ! -d "output" ]; then
  echo "Directory for output does not exist create it!"
  mkdir output
fi

echo "Running Poisson script"
python3 poisson.py > output/poisson.txt

echo "Running LU decomposition script"
python3 lu_decomp.py > output/lu_decomp.txt

echo "Running Neville's algorithm script"
python3 neville.py > output/neville.txt

echo "Running LU with iterations script"
python3 iteration_lu.py > output/iteration_lu.txt

echo "Running timing script"
python3 time_1.py > output/time_1.txt

echo "Generating the pdf"

pdflatex TeX/report_1.tex
bibtex TeX/report_1.aux
pdflatex TeX/report_1.tex
pdflatex TeX/report_1.tex

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

