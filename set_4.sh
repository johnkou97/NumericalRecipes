#!/bin/bash
echo "Run Handin assignment 4"

echo "Creating the plotting and output directories"

if [ ! -d "plots" ]; then
  echo "Directory for plots does not exist create it!"
  mkdir plots
fi

if [ ! -d "output" ]; then
  echo "Directory for output does not exist create it!"
  mkdir output
fi

echo "Running initial script"
python3 initial.py

echo "Running leapfrog script"
python3 leapfrog.py

echo "Running euler script"
python3 euler.py

echo "Running generate script"
python3 generate.py

echo "Running fft script"
python3 fft.py 

echo "Running prepare script"
python3 prepare.py

echo "Running classification script"
python3 classification.py 

echo "Running evaluation script"
python3 evaluation.py > output/evaluation.txt

echo "Generating the pdf"

pdflatex TeX/report_4.tex
bibtex TeX/report_4.aux
pdflatex TeX/report_4.tex
pdflatex TeX/report_4.tex

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