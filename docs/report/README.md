# Formal Report (LaTeX)

Main file:
- `docs/report/main.tex`

## Build prerequisites

Install TeX packages (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra texlive-pictures
```

## Compile

```bash
cd docs/report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output:
- `docs/report/main.pdf`
