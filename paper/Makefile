
default: all

all: article

article:
	$(MAKE) articletex
	$(MAKE) bib
	$(MAKE) articletex
	$(MAKE) articletex

articletex:
	pdflatex article.tex

bib:
	bibtex article.aux


anon:
	$(MAKE) articleanontex
	$(MAKE) bibanon
	$(MAKE) articleanontex
	$(MAKE) articleanontex

articleanontex:
	pdflatex article_anon.tex

bibanon:
	bibtex article_anon.aux
