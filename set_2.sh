#!/bin/bash
echo "Run Handin assignment 2"

echo "Creating the plotting and output directories"

if [ ! -d "plots" ]; then
  echo "Directory for plots does not exist create it!"
  mkdir plots
fi

if [ ! -d "output" ]; then
  echo "Directory for output does not exist create it!"
  mkdir output
fi

echo "Running open_int script"
python3 open_int.py > output/open_int.txt

echo "Running distr script"
python3 distr.py 

echo "Running sort script"
python3 sort.py

echo "Running deriv script"
python3 deriv.py > output/deriv.txt

echo "Running root1 script"
python3 root1.py > output/root1.txt

echo "Running root2 script"
python3 root2.py > output/root2.txt

echo "Running time script"
python3 time_2.py > output/time_2.txt

echo "Generating the pdf"

pdflatex TeX/report_2.tex 
bibtex report_2.aux 
pdflatex TeX/report_2.tex 
pdflatex TeX/report_2.tex 

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