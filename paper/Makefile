
default: all

all:
	$(MAKE) article
	$(MAKE) bib
	$(MAKE) article
	$(MAKE) article

article:
	pdflatex article.tex

bib:
	bibtex article.aux

