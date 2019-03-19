BIBLIOGRAPHY_SOURCE = bibliography.bib

PAPER = main
PAPER_SOURCE = $(PAPER).tex
PAPER_AUX = $(PAPER).aux
PAPER_TARGET = $(PAPER).pdf

RED=`echo '\033[0;31m'`
YELLOW=`echo '\033[0;33m'`
BLUE=`echo '\033[0;34m'`
NOCOLOR=`echo '\033[m'`
BOLD=`echo '\033[1m'`
NORMAL=`echo '\033[0m'`

COLORIZE = \
		   sed "s/Overfull/${RED}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/Underfull/${YELLOW}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/[0-9]*--[0-9]*/${BLUE}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/LaTeX Warning/${YELLOW}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/! LaTeX Error/${RED}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/! Undefined control sequence/${RED}${BOLD}&${NORMAL}${NOCOLOR}/g" | \
		   sed "s/l\.[0-9]*/${BLUE}${BOLD}&${NORMAL}${NOCOLOR}/g"

$(PAPER_TARGET): $(PAPER_SOURCE) $(BIBLIOGRAPHY_SOURCE)
	@echo ">>> PRERUN PDFLATEX <<<"
	@echo ""
	pdflatex -shell-escape -interaction=nonstopmode $(PAPER_SOURCE) 1> /dev/null 2> /dev/null
	@echo ""
	@echo ">>> BIBTEX <<<"
	@echo ""
	bibtex $(PAPER_AUX)
	@echo ""
	@echo ">>> GLOSSARIES <<<"
	@echo ""
	makeglossaries $(PAPER)
	@echo ""
	@echo ">>> PRERUN PDFLATEX <<<"
	@echo ""
	pdflatex -shell-escape -interaction=nonstopmode $(PAPER_SOURCE) 1> /dev/null 2> /dev/null
	@echo ""
	@echo ">>> PRERUN PDFLATEX <<<"
	@echo ""
	pdflatex -shell-escape -interaction=nonstopmode $(PAPER_SOURCE) 1> /dev/null 2> /dev/null
	@echo ""
	@echo ">>> PDFLATEX <<<"
	@echo ""
	pdflatex -shell-escape -interaction=nonstopmode $(PAPER_SOURCE) | $(COLORIZE)

all: $(PAPER_TARGET)

.PHONY: clean
clean:
	@rm -f *.out *.aux *.log *.pdf *.bbl *.blg *.dvi *.toc *.lof *.lol *.lot *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.xdy *.nco *.ncg *.ncs *.bcf 
